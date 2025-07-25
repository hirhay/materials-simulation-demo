import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="簡易バンド構造ビューア", layout="wide")

st.title("金属と半導体のバンド構造を体験しよう")

# 物理定数
kB = 8.617333e-5  # eV/K

# パラメータ設定UI
material = st.selectbox("材料を選ぶ", ["Si (半導体)", "Cu (金属)"])
T = st.slider("温度 T [K]", 0, 800, 300, step=25)
doping = st.slider("ドーピング（p型 ← 0 → n型）", -1.0, 1.0, 0.0, 0.1)

# k点
Nk = 400
k = np.linspace(-np.pi, np.pi, Nk)

# モデルパラメータ
t = 1.0
if material.startswith("Cu"):
    # 金属：1バンド
    Eg = 0.0
    E_bands = [ -2*t * np.cos(k) ]  # list of arrays
    labels = ["バンド"]
    # フェルミ準位（dopingで少し上下させる）
    mu0 = 0.0
    mu = mu0 + 0.5 * doping  # 簡易シフト
else:
    # Si: ギャップEgと2バンド
    Eg = 1.1  # eV（簡易）
    Ev = -Eg/2 - 2*t*np.cos(k)
    Ec = +Eg/2 + 2*t*np.cos(k)
    E_bands = [Ev, Ec]
    labels = ["価電子帯", "伝導帯"]
    # mid-gap を基準にドーピングで化学ポテンシャル移動
    mu = 0.0 + 0.5*doping*Eg  # -Eg/2～+Eg/2 付近を遷移

# Fermi-Dirac 分布
def fermi(E, mu, T):
    if T == 0:
        return (E < mu).astype(float)
    beta = 1.0 / (kB*T)
    return 1.0 / (np.exp((E - mu)*beta) + 1.0)

occupations = [fermi(E, mu, T) for E in E_bands]

# キャリア密度（超簡易）：半導体なら伝導帯占有 - 価電子帯空孔
if material.startswith("Si"):
    n_e = np.trapz(occupations[1], k)/(2*np.pi)     # 電子数(相対)
    p_h = np.trapz(1 - occupations[0], k)/(2*np.pi) # 正孔数(相対)
    carrier_text = f"電子密度 ~ {n_e:.3f}, 正孔密度 ~ {p_h:.3f}"
else:
    n_tot = np.trapz(occupations[0], k)/(2*np.pi)
    carrier_text = f"占有電子数(相対) ~ {n_tot:.3f}"

# プロット
fig = go.Figure()
colors = ["#4E79A7", "#F28E2B"]

for i, (E, occ, lab) in enumerate(zip(E_bands, occupations, labels)):
    # 線の透明度で占有を表現（0→薄い,1→濃い）
    alpha = 0.2 + 0.8*occ  # 0.2～1.0
    # 線を細切れに描く：占有度ごとにcolor RGBA
    for j in range(Nk-1):
        fig.add_trace(go.Scatter(
            x=k[j:j+2], y=E[j:j+2],
            mode="lines",
            line=dict(color=f"rgba({78 if i==0 else 242}, {121 if i==0 else 142}, {167 if i==0 else 43}, {alpha[j]:.3f})", width=3),
            showlegend=False
        ))
    # 凡例用ダミー
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color=colors[i], width=3),
                             name=lab))

# フェルミ準位
fig.add_hline(y=mu, line=dict(color="black", dash="dash"), annotation_text="μ (Fermi)", annotation_position="top left")

# ギャップ表示
if Eg > 0:
    fig.add_shape(type="rect",
                  x0=k[0], x1=k[-1], y0=-Eg/2, y1=+Eg/2,
                  fillcolor="lightgrey", opacity=0.15, line_width=0)

fig.update_layout(
    xaxis_title="k（1次元ブリルアンゾーン）",
    yaxis_title="エネルギー E [eV]",
    margin=dict(l=10, r=10, t=30, b=10),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("バンドギャップ Eg", f"{Eg:.2f} eV")
with col2:
    st.metric("化学ポテンシャル μ", f"{mu:.2f} eV")

st.write(carrier_text)
st.caption("※単純化したモデル。実際のSiは間接ギャップ、Cuは複雑な多バンドですが、原理理解のために簡約化しています。")
