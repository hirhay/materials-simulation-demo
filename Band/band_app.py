import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="金属と半導体の電子状態", layout="wide")

st.title("金属と半導体の電子状態を体験しよう")
st.write("材料や温度を変えると、電子がどのようにエネルギー準位を占有するかが変化します。特に半導体では、温度を上げると電子が伝導帯にジャンプし、電気が流れるようになります。")

# --- サイドバー：パラメータ設定 ---
st.sidebar.header("パラメータ設定")
material = st.sidebar.selectbox("材料を選ぶ", ["Si (半導体)", "Cu (金属)"])
T = st.sidebar.slider("温度 T [K]", 0, 800, 300, step=10)
doping = st.sidebar.slider("ドーピング（p型 ← 0 → n型）", -1.0, 1.0, 0.0, 0.1)

# --- 物理定数とモデル ---
kB = 8.617333e-5  # eV/K
t = 1.0  # ホッピングパラメータ

# 1次元のk点
Nk = 500
k = np.linspace(-np.pi, np.pi, Nk)

# 材料に応じたバンド構造と化学ポテンシャルの設定
if material.startswith("Cu"):
    # 金属：1バンドモデル（cosモデルでバンド端のDOSが高いことを示す）
    Eg = 0.0
    E_bands = [-2 * t * np.cos(k)]
    mu = 0.0 + 0.5 * doping
    E_range = (-3, 3)
else:
    # 半導体：より現実に近い放物線モデル
    Eg = 1.1  # Siのバンドギャップ (eV)
    parabolic_const = t / (np.pi**2) * 2.0 # バンドの曲率を調整
    Ev = -Eg/2 - parabolic_const * k**2  # 価電子帯
    Ec = +Eg/2 + parabolic_const * k**2  # 伝導帯
    E_bands = [Ev, Ec]
    mu = 0.0 + 0.6 * doping * Eg
    E_range = (-2.5, 2.5)

# --- 計算関数 ---
def fermi(E, mu, T):
    """フェルミ・ディラック分布関数"""
    if T == 0:
        return (E < mu).astype(float)
    beta = 1.0 / (kB * T)
    arg = np.clip((E - mu) * beta, -500, 500)
    return 1.0 / (np.exp(arg) + 1.0)

# エネルギーグリッド上で各種量を計算
n_bins = 400
E_grid = np.linspace(E_range[0], E_range[1], n_bins)

# 状態密度(DOS)を計算
dos_total = np.zeros_like(E_grid)
for E_band in E_bands:
    hist, _ = np.histogram(E_band, bins=n_bins, range=E_range)
    dos_total += hist

# ガウシアンフィルタで平滑化（sigmaを小さくしてギャップを明確に）
dos_total = gaussian_filter1d(dos_total.astype(float), sigma=1.5)
if dos_total.max() > 0:
    dos_total = dos_total / dos_total.max() * 100 # 正規化

# フェルミ分布と占有されている状態密度
f_dist = fermi(E_grid, mu, T)
dos_occupied = dos_total * f_dist

# --- 描画 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("状態密度 (DOS)")
    fig_dos = go.Figure()

    # バンドギャップの領域をハイライト
    if material.startswith("Si"):
        fig_dos.add_hrect(y0=-Eg/2, y1=Eg/2,
                          fillcolor="rgba(128, 128, 128, 0.15)",
                          line_width=0,
                          annotation_text="バンドギャップ",
                          annotation_position="top left",
                          annotation_font_size=12)

    # 全状態密度
    fig_dos.add_trace(go.Scatter(x=dos_total, y=E_grid, name="全状態 (電子のイス)",
                                 line=dict(color="lightgrey", width=2)))
    # 占有状態密度
    fig_dos.add_trace(go.Scatter(x=dos_occupied, y=E_grid, name="占有状態 (座っている電子)",
                                 fill='tozerox', line=dict(color="#4E79A7"),
                                 fillcolor="rgba(78, 121, 167, 0.6)"))
    # フェルミ準位
    fig_dos.add_hline(y=mu, line=dict(color="black", dash="dash"),
                      annotation_text="μ (化学ポテンシャル)", annotation_position="bottom right")
    fig_dos.update_layout(xaxis_title="状態の数 (任意単位)", yaxis_title="エネルギー E [eV]",
                          margin=dict(l=10, r=10, t=40, b=10), height=500, showlegend=True,
                          legend=dict(x=0.05, y=0.95))
    fig_dos.update_yaxes(range=E_range)
    st.plotly_chart(fig_dos, use_container_width=True)

with col2:
    st.subheader("フェルミ分布関数 f(E)")
    fig_fermi = go.Figure()
    fig_fermi.add_trace(go.Scatter(x=f_dist, y=E_grid, name="f(E)",
                                   line=dict(color="#F28E2B", width=3)))
    # フェルミ準位
    fig_fermi.add_hline(y=mu, line=dict(color="black", dash="dash"),
                        annotation_text="μ", annotation_position="bottom right")
    fig_fermi.update_layout(xaxis_title="占有確率", yaxis_title="エネルギー E [eV]",
                            margin=dict(l=10, r=10, t=40, b=10), height=500,
                            xaxis_range=[-0.05, 1.05])
    fig_fermi.update_yaxes(range=E_range)
    st.plotly_chart(fig_fermi, use_container_width=True)

# --- サマリー表示 ---
st.metric("化学ポテンシャル μ", f"{mu:.3f} eV")

if material.startswith("Si"):
    # 伝導帯の電子数と価電子帯の正孔数を計算（簡易）
    cond_mask = E_grid > (Eg/2)
    vale_mask = E_grid < (-Eg/2)

    n_e = np.trapz(dos_occupied[cond_mask], E_grid[cond_mask])
    p_h = np.trapz(dos_total[vale_mask] * (1 - f_dist[vale_mask]), E_grid[vale_mask])
    carrier_text = f"伝導電子密度 (相対値): {n_e/100:.3f}  |  正孔密度 (相対値): {p_h/100:.3f}"
    st.metric("バンドギャップ Eg", f"{Eg:.2f} eV")
else:
    carrier_text = "金属は常に多数の伝導電子を持つ"

st.info(carrier_text)
st.caption("※単純化した1次元モデルです。実際のSiは間接ギャップ、Cuは複雑なバンド構造を持ちますが、金属と半導体の本質的な違いを理解するために簡約化しています。")