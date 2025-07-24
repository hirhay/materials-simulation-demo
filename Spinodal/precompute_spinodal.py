# precompute_compare.py
import numpy as np
from pathlib import Path

# --- 共通パラメータ ---
Nx = Ny = 256
dx = 1.0
dt = 1e-2
n_steps = 1200
save_int = 20

A = 1.0
kappa = 1.0
M = 1.0

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi*np.fft.fftfreq(Ny, d=dx)
k2 = np.add.outer(kx**2, ky**2)

def run_simulation(mu_func, label):
    c = 0.01*(np.random.rand(Nx,Ny)-0.5)    # 小揺らぎ
    c_hat = np.fft.fftn(c)
    frames, times = [c.copy()], [0.]
    for step in range(1, n_steps+1):
        c_real = np.fft.ifftn(c_hat).real
        mu_real = mu_func(c_real) - kappa*np.fft.ifftn(k2*c_hat).real
        mu_hat  = np.fft.fftn(mu_real)
        denom = 1 + dt*M*kappa*k2**2
        c_hat = (c_hat - dt*M*k2*mu_hat) / denom
        if step % save_int == 0:
            frames.append(np.fft.ifftn(c_hat).real.copy())
            times.append(step*dt)
    np.save(data_dir/f"conc_{label}.npy", np.array(frames, dtype=np.float32))
    return times

# --- 不安定系 (スピノーダル) ---
times = run_simulation(lambda c: A*(c**3 - c), "unstable")

# --- 安定系 (単井戸)  ---
run_simulation(lambda c: A*c, "stable")

np.save(data_dir/"time.npy", np.array(times, dtype=np.float32))
print("Finished.  Frames:", len(times))
