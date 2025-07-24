# app_compare.py
import streamlit as st, numpy as np, plotly.express as px
from pathlib import Path

st.set_page_config(page_title="スピノーダル vs 安定系", layout="wide")

data_dir = Path("data")
conc_un  = np.load(data_dir/"conc_unstable.npy")
conc_st  = np.load(data_dir/"conc_stable.npy")
times    = np.load(data_dir/"time.npy")
Nframe   = len(times)

idx = st.slider("シミュレーションフレーム", 0, Nframe-1, 0)
c_un, c_st = conc_un[idx], conc_st[idx]
t = times[idx]

def show_img(arr, title):
    fig = px.imshow(arr, origin="lower", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu", aspect="equal")
    fig.update_layout(coloraxis_showscale=False,
                      margin=dict(l=0,r=0,t=30,b=0), height=400,
                      title=title)
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    show_img(c_un, "不安定系：スピノーダル分解")
with right:
    show_img(c_st, "安定系：分解せず")

st.markdown(f"**時間 = {t:.2f}**  (任意単位)")
st.markdown("""
- 左はまだら模様が成長し **ドメインが粗大化**。  
- 右は初期のノイズが **時間とともに平坦化** していく様子が比較できます。
""")
