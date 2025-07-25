# precompute_spinodal.py
import numpy as np
from pathlib import Path

# =============================================================================
# 1. 物理パラメータと単位系の設定
# =============================================================================
A_phys = 1e9
kappa_phys = 1e-10
M_phys = 1.25e-18
L_unit = 1e-9

A = 1.0
M = 1.0
# NOTE: kappaの値を大きくして、スピノーダル分解の模様をさらに粗大化させる
kappa = 256.0 # 無次元の界面エネルギー (16.0 -> 256.0)

time_scale = (L_unit**2) / (M_phys * A_phys)

# =============================================================================
# 2. シミュレーションの数値設定
# =============================================================================
Nx = Ny = 256
dx = 1.0
dt = 1e-2

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
k2 = np.add.outer(kx**2, ky**2)

# =============================================================================
# 3. シミュ��ーション実行関数
# =============================================================================
def run_simulation(c_initial, mu_func, label, n_steps, save_points, damping_factor=0.0):
    """ 汎用的なカーン・ヒリアード方程式のソルバー """
    c_hat = np.fft.fftn(c_initial)
    frames, times = [c_initial.copy()], [0.]
    
    noise_filter = np.exp(-damping_factor * k2 * dt)
    save_idx = 0

    for step in range(1, n_steps + 1):
        c_real = np.fft.ifftn(c_hat).real
        mu_real = mu_func(c_real) - kappa * np.fft.ifftn(k2 * c_hat).real
        mu_hat = np.fft.fftn(mu_real)
        denom = 1 + dt * M * kappa * k2**2
        c_hat = (c_hat - dt * M * k2 * mu_hat) / denom
        
        if damping_factor > 0:
            c_hat *= noise_filter

        if save_idx < len(save_points) and step == save_points[save_idx]:
            frames.append(np.fft.ifftn(c_hat).real.copy())
            times.append(step * dt)
            save_idx += 1
            
    np.save(data_dir / f"conc_{label}.npy", np.array(frames, dtype=np.float32))
    return times

# =============================================================================
# 4. 各ケースの計算実行
# =============================================================================

# --- ケース1: 不安定系 (スピノーダル分解) ---
# フレーム数を核形成シミュレーションと厳密に合わせる (合計61フレーム)
n_steps_unstable = 48000
# ステージ1：潜伏期間 (超粗く)
save_points_s1 = np.array([500, 1000], dtype=int) # 2 フレーム
# ステージ2：分解の瞬間 (超密に)
save_points_s2 = np.arange(1200, 1801, 25, dtype=int) # 25 フレーム
# ステージ3：初期成長期 (密に)
save_points_s3 = np.arange(1850, 5001, 180, dtype=int) # 18 フレーム
# ステージ4：後期・粒成長期 (粗く)
save_points_s4 = np.linspace(5200, n_steps_unstable, 15, dtype=int) # 15 フレーム
save_points_unstable = np.unique(np.concatenate([save_points_s1, save_points_s2, save_points_s3, save_points_s4]))

c0_unstable = 0.01 * (np.random.rand(Nx, Ny) - 0.5)
mu_func_unstable = lambda c: A * (c**3 - c)
times = run_simulation(c0_unstable, mu_func_unstable, "unstable", n_steps_unstable, save_points_unstable)
print(f"スピノーダル分解の計算完了。フレーム数: {len(times)}")


# --- ケース2: 準安定系 (核形成・成長) ---
# こちらは変更なし
n_steps_nucleation = 48000
save_points_nucleation = np.linspace(0, n_steps_nucleation, 61, dtype=int)[1:]

c0_nucleation = -0.4 + 0.01 * (np.random.rand(Nx, Ny) - 0.5)
mu_func_nucleation = lambda c: A * (c**3 - c)
run_simulation(c0_nucleation, mu_func_nucleation, "nucleation", n_steps_nucleation, save_points_nucleation, damping_factor=0.05)
print("核形成・成長の計算完了。")

# --- 最終データの保存 ---
np.save(data_dir / "time.npy", np.array(times, dtype=np.float32))
np.save(data_dir / "phys_params.npy", np.array([time_scale, L_unit]))
print("すべての計算が完了しました。")




