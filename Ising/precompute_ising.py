"""
3 種類のイジングモデル (Fe, Ni, Gd 想定) を温度走査し、
・スピン配置 PNG
・磁化 csv
を materials/{Fe,Ni,Gd}/ に保存する。
"""

import os, csv
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------- パラメータ ----------
L = 64                     # 格子サイズ (L×L)
steps_eq = 900           # 各温度での平衡化ステップ
steps_meas = 200         # 測定ステップ
temps = np.arange(0.0, 1201, 5)  # 0–1200 K を 5 K 刻み
kB = 1.0                   # Boltzマン定数: 規格化

# 材料ごとの (名前, 実 Tc[K])
mats = [("Fe", 1043), ("Ni", 627), ("Gd", 293)]

# ---------- Ising コア ----------
@njit
def metropolis_step(spins, beta):
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)
        s = spins[i, j]
        nn = spins[(i+1)%L,j] + spins[(i-1)%L,j] + spins[i,(j+1)%L] + spins[i,(j-1)%L]
        dE = 2 * s * nn
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = -s

def simulate(beta, steps_eq, steps_meas, spins):
    for _ in range(steps_eq):
        metropolis_step(spins, beta)
    m_sum = 0.0
    for _ in range(steps_meas):
        metropolis_step(spins, beta)
        m_sum += np.abs(spins.mean())
    return m_sum / steps_meas, spins.copy()

# ---------- ループ ----------
for name, Tc_real in mats:
    # 実 Tc を 2D Ising の Tc=2.269J/kB へ線形スケールし J を決定
    J = Tc_real / 2.269
    dest = f"materials/{name}"
    os.makedirs(dest, exist_ok=True)

    spins = np.ones((L, L), dtype=np.int8)
    csv_path = os.path.join(dest, "magnetization.csv")
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["T_K", "M_abs"])

        for T in temps:
            beta = 1 / (kB * T / J) if T > 0 else 1e6
            M, frame = simulate(beta, steps_eq, steps_meas, spins)

            # 画像保存（赤:↓ 青:↑）
            #cmap = plt.get_cmap("bwr")
            cmap = ListedColormap([ "#4E79A7",   # ダウンスピン → 落ち着いた青
                                    "#F28E2B"])  # アップスピン → ソフトオレンジ
            plt.imsave(os.path.join(dest, f"{int(T):04}.png"),
                       (frame+1)/2, cmap=cmap, vmin=0, vmax=1)

            writer.writerow([T, M])
            print(f"{name}: T={T:4.0f} K  M={M:.3f}")
