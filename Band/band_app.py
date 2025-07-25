import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="金属と半導体の電子状態", layout="wide")

st.title("金属と半導体の電子状態を体験しよう")
st.write("材料や温度、不純物（ドーピング）を変えると電子の状態がどう変わるか見てみましょう。特に半導体（Si）では、右のグラフで温度上昇に伴いキャリア（電子・正孔）がどう増えるかに注目してください。")

# --- サイドバー：パラメータ設定 ---
st.sidebar.header("パラメータ設定")
material = st.sidebar.selectbox("材料を選ぶ", ["Si (半導体)", "Cu (金属)"])
T_slider = st.sidebar.slider("温度 T [K]", 1, 800, 300, step=10)
doping = st.sidebar.slider("ドーピング（p型 ← 0 → n型）", -1.0, 1.0, 0.0, 0.1)

# --- 物理定数とモデル ---
kB = 8.617333e-5  # eV/K
Eg = 1.1  # Siのバンドギャップ (eV)

# --- 計算関数 ---
@st.cache_data
def calculate_carriers_vs_temp(doping_level):
    """指定されたドーピングレベルで、全温度範囲のキャリア濃度を計算する"""
    T_range = np.linspace(1, 800, 100)
    n_e_vs_T = np.zeros_like(T_range)
    p_h_vs_T = np.zeros_like(T_range)

    E_range_dos = (-2.5, 2.5)
    E_grid_dos = np.linspace(E_range_dos[0], E_range_dos[1], 400)
    
    band_width = 1.4
    center_v = -Eg/2 - band_width/2
    center_c = Eg/2 + band_width/2
    dos_v = np.sqrt(np.maximum(0, 1 - ((E_grid_dos - center_v) / (band_width/2))**2))
    dos_c = np.sqrt(np.maximum(0, 1 - ((E_grid_dos - center_c) / (band_width/2))**2))
    dos_total = (dos_v + dos_c) * 100
    
    for i, T in enumerate(T_range):
        beta = 1.0 / (kB * T)
        mu_intrinsic = 0.15 * (T / 800.0)**2 * Eg
        mu_doping = 0.6 * doping_level * Eg
        mu = mu_intrinsic + mu_doping
        
        arg = np.clip((E_grid_dos - mu) * beta, -500, 500)
        f_dist = 1.0 / (np.exp(arg) + 1.0)
        
        dos_occupied = dos_total * f_dist
        
        cond_mask = E_grid_dos >= Eg/2
        vale_mask = E_grid_dos <= -Eg/2
        
        n_e_vs_T[i] = np.trapezoid(dos_occupied[cond_mask], E_grid_dos[cond_mask])
        p_h_vs_T[i] = np.trapezoid(dos_total[vale_mask] * (1 - f_dist[vale_mask]), E_grid_dos[vale_mask])
        
    return T_range, n_e_vs_T, p_h_vs_T

# --- メインの描画ロジック ---
col1, col2 = st.columns(2)

# --- 左側のプロット（状態密度） ---
with col1:
    st.subheader("状態密度 (DOS)")
    
    E_range_dos = (-2.5, 2.5)
    E_grid_dos = np.linspace(E_range_dos[0], E_range_dos[1], 400)
    
    if material.startswith("Cu"):
        k = np.linspace(-np.pi, np.pi, 500)
        E_band = -2 * 1.0 * np.cos(k)
        dos_total, bins = np.histogram(E_band, bins=400, range=(-3,3))
        dos_total = gaussian_filter1d(dos_total.astype(float), sigma=2.0)
        E_grid_dos = (bins[:-1] + bins[1:]) / 2
        mu = 0.0 + 0.5 * doping
    else: # Si
        band_width = 1.4
        center_v = -Eg/2 - band_width/2
        center_c = Eg/2 + band_width/2
        dos_v = np.sqrt(np.maximum(0, 1 - ((E_grid_dos - center_v) / (band_width/2))**2))
        dos_c = np.sqrt(np.maximum(0, 1 - ((E_grid_dos - center_c) / (band_width/2))**2))
        dos_total = (dos_v + dos_c)
        mu_intrinsic = 0.15 * (T_slider / 800.0)**2 * Eg
        mu_doping = 0.6 * doping * Eg
        mu = mu_intrinsic + mu_doping

    if np.max(dos_total) > 0:
        dos_total = dos_total / np.max(dos_total) * 100

    if T_slider > 0:
        beta = 1.0 / (kB * T_slider)
        arg = np.clip((E_grid_dos - mu) * beta, -500, 500)
        f_dist = 1.0 / (np.exp(arg) + 1.0)
    else:
        f_dist = (E_grid_dos < mu).astype(float)
        
    dos_occupied = dos_total * f_dist

    fig_dos = go.Figure()
    if material.startswith("Si"):
        fig_dos.add_hrect(y0=-Eg/2, y1=Eg/2, fillcolor="rgba(128,128,128,0.15)", line_width=0,
                          annotation_text="バンドギャップ", annotation_position="top left")
    fig_dos.add_trace(go.Scatter(x=dos_total, y=E_grid_dos, name="全状態", line=dict(color="lightgrey", width=2)))
    fig_dos.add_trace(go.Scatter(x=dos_occupied, y=E_grid_dos, name="占有状態", fill='tozerox', line=dict(color="#4E79A7")))
    fig_dos.add_hline(y=mu, line=dict(color="black", dash="dash"), annotation_text="μ", annotation_position="bottom right")
    fig_dos.update_layout(xaxis_title="状態の数 (任意単位)", yaxis_title="エネルギー E [eV]", margin=dict(l=10,r=10,t=40,b=10), height=500, legend=dict(x=0.05, y=0.95))
    st.plotly_chart(fig_dos, use_container_width=True)

# --- 右側のプロット（キャリア濃度） ---
with col2:
    st.subheader("キャリア濃度 vs 温度")
    fig_carrier = go.Figure()
    if material.startswith("Si"):
        T_range, n_e, p_h = calculate_carriers_vs_temp(doping)
        fig_carrier.add_trace(go.Scatter(x=T_range, y=n_e, name="電子 (n_e)", line=dict(color="#F28E2B", width=3)))
        fig_carrier.add_trace(go.Scatter(x=T_range, y=p_h, name="正孔 (p_h)", line=dict(color="#4E79A7", width=3)))
        fig_carrier.update_yaxes(type="log", range=[-1, 2.5], title_text="キャリア濃度 (任意単位)")
        fig_carrier.add_vline(x=T_slider, line=dict(color="black", dash="dash"), annotation_text=f"T = {T_slider} K")
    else: # 金属の場合
        fig_carrier.add_annotation(text="金属は常に高濃度のキャリアを持つ", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
        fig_carrier.update_yaxes(range=[0, 1]) # ダミーの範囲

    fig_carrier.update_layout(xaxis_title="温度 T [K]", margin=dict(l=10,r=10,t=40,b=10), height=500, legend=dict(x=0.05, y=0.95))
    st.plotly_chart(fig_carrier, use_container_width=True)

# --- サマリー ---
st.metric("化学ポテンシャル μ", f"{mu:.3f} eV")
if material.startswith("Si"):
    st.metric("バンドギャップ Eg", f"{Eg:.2f} eV")
st.caption("※単純化したモデルです。実際のSiは間接ギャップ、Cuは複雑なバンド構造を持ちますが、金属と半導体の本質的な違いを理解するために簡約化しています。")