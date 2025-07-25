# precompute_melting.py
import os, numpy as np
from pathlib import Path

# ---------------- MD パラメータ ----------------
Ncell = 6            # 立方格子の単位胞数 (N^3 原子)
a = 1.0              # 格子定数
mass = 1.0           # 原子質量 (LJ 単位)
dt = 0.001           # タイムステップ (小さくして安定化)
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
pos_initial = pos.copy()  # t=0 の初期位置を保存

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
        rij -= boxL * np.rint(rij / boxL)
        dist2 = np.sum(rij**2, axis=1)
        mask = (dist2 < rc2) & (dist2 > 0)
        rij = rij[mask]; dist2 = dist2[mask]
        if dist2.size == 0:
            continue
        inv_r2 = sigma**2 / dist2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        f_scalar = 24 * epsilon * (2*inv_r12 - inv_r6) / dist2
        fij = f_scalar[:, np.newaxis] * rij
        forces[i] += np.sum(fij, axis=0)
        forces[i+1:][mask] -= fij
        pot += 4*epsilon * np.sum(inv_r12 - inv_r6)
    return forces, pot

# ------ 動径分布関数 g(r) 計算 ------
def compute_rdf(positions, boxL, n_bins=50, r_max=None):
    if r_max is None:
        r_max = boxL / 2.0
    
    n_atoms = len(positions)
    rho = n_atoms / boxL**3  # 数密度
    
    distances = []
    for i in range(n_atoms):
        rij = positions[i+1:] - positions[i]
        rij -= boxL * np.rint(rij / boxL)
        dist = np.linalg.norm(rij, axis=1)
        distances.extend(dist)
    
    distances = np.array(distances)
    
    # ヒストグラム作成
    hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
    r = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    dr = r[1] - r[0]
    
    # 正規化
    shell_volume = 4.0 * np.pi * r**2 * dr
    n_ideal = shell_volume * rho
    
    # N(N-1)/2 で割るべきだが、ここでは N で割ることでペア数を考慮
    g_r = hist / (n_ideal * n_atoms * 0.5)
    
    return r, g_r

# ------ データ保存用配列 ------
snapshots = []
msd_list   = []
rdf_list = []

# ------ MD ループ ------
print("Starting MD simulation...")
for i, T in enumerate(temps):
    for step in range(steps_per_T):
        forces, pot = compute_forces(pos)
        vel += 0.5 * forces / mass * dt
        vel = np.clip(vel, -100.0, 100.0)

        pos += vel * dt
        pos %= boxL

        forces, pot = compute_forces(pos)
        vel += 0.5 * forces / mass * dt
        vel = np.clip(vel, -100.0, 100.0)

        kin = 0.5 * mass * np.sum(vel**2)
        current_T = (2*kin)/(3*Natoms)
        vel *= np.sqrt(T / (current_T + 1e-9))

        if step % record_interval == 0:
            snapshots.append(pos.copy())
            
            displacement = pos - pos_initial
            displacement -= boxL * np.rint(displacement / boxL)
            squared_disp = np.sum(displacement**2, axis=1)
            msd_list.append(np.mean(squared_disp))
            
            r_axis, g_r = compute_rdf(pos, boxL)
            rdf_list.append(g_r)
    
    print(f"Finished T = {T:.2f} ({i+1}/{len(temps)})")

print(f"Saved {len(snapshots)} frames")

np.save(data_dir / "frames.npy", np.array(snapshots))
np.save(data_dir / "temps.npy", temps.repeat(steps_per_T//record_interval))
np.save(data_dir / "msd.npy",  np.array(msd_list))
np.save(data_dir / "rdfs.npy", np.array(rdf_list))
np.save(data_dir / "rdf_r_axis.npy", r_axis)

print("Precomputation finished.")
print(f"Data saved in '{data_dir}' directory.")
print("To run the demo, execute: streamlit run app.py")