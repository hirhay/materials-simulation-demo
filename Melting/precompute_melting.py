# precompute_melting.py
import os, numpy as np
from pathlib import Path

# ---------------- MD パラメータ ----------------
Ncell = 6            # 立方格子の単位胞数 (N^3 原子)
a = 1.0              # 格子定数
mass = 1.0           # 原子質量 (LJ 単位)
dt = 0.002           # タイムステップ
steps_per_T = 400    # 各温度での MD ステップ
record_interval = 50 # スナップショット間隔
temps = np.linspace(0.2, 2.0, 40)  # 融解(~1.2)をまたぐ昇温

epsilon = 1.0; sigma = 1.0; rc = 2.5 * sigma
boxL = Ncell * a
rc2 = rc**2
# ----------------------------------------------

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# ------ 格子初期配置 ------
coords = [(i, j, k) for i in range(Ncell)
                      for j in range(Ncell)
                      for k in range(Ncell)]
pos = np.array(coords, dtype=np.float64) * a   # [N,3]
Natoms = pos.shape[0]

# ------ 速度初期化 ------
def maxwell_vel(T):
    return np.random.normal(scale=np.sqrt(T/mass), size=(Natoms, 3))

vel = maxwell_vel(temps[0])

# ------ 力計算 (Lennard-Jones) ------
def compute_forces(positions):
    forces = np.zeros_like(positions)
    pot = 0.0
    for i in range(Natoms - 1):
        ri = positions[i]
        rij = positions[i+1:] - ri
        # 最近接画像法 (周期境界条件)
        rij -= boxL * np.rint(rij / boxL)
        dist2 = np.sum(rij**2, axis=1)
        mask = dist2 < rc2
        rij = rij[mask]; dist2 = dist2[mask]
        if dist2.size == 0:
            continue
        inv_r2 = sigma**2 / dist2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        fij = 24 * epsilon * (2*inv_r12 - inv_r6) * sigma**-2 * rij / dist2[:,None]
        forces[i] += np.sum(fij, axis=0)
        forces[i+1:][mask] -= fij
        pot += 4*epsilon * np.sum(inv_r12 - inv_r6)
    return forces, pot

# ------ データ保存用配列 ------
snapshots = []
msd_list   = []

# ------ MD ループ ------
for T in temps:
    # ベルレ積分器
    for step in range(steps_per_T):
        forces, pot = compute_forces(pos)
        vel += 0.5 * forces / mass * dt
        pos += vel * dt
        pos %= boxL  # 周期境界
        forces, pot = compute_forces(pos)
        vel += 0.5 * forces / mass * dt

        # 瞬時温度から rescale (単純な加熱サーモスタット)
        kin = 0.5 * mass * np.sum(vel**2)
        current_T = (2*kin)/(3*Natoms)
        vel *= np.sqrt(T / current_T)

        # スナップショット
        if step % record_interval == 0:
            snapshots.append(pos.copy())
            # MSD: 固体 → 小, 液体 → 大
            disp = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
            msd_list.append(np.mean(disp**2))

print(f"saved {len(snapshots)} frames")

np.save(data_dir / "frames.npy", np.array(snapshots))
np.save(data_dir / "temps.npy", temps.repeat(steps_per_T//record_interval))
np.save(data_dir / "msd.npy",  np.array(msd_list))
