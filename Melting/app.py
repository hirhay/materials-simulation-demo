# app.py
import streamlit as st
import numpy as np
import plotly.graph_objs as go
from pathlib import Path

st.set_page_config(page_title="原子めがねで見る融解", layout="wide")

data_dir = Path("data")
frames = np.load(data_dir / "frames.npy")      # [Nframe, Natom, 3]
temps  = np.load(data_dir / "temps.npy")       # [Nframe]
msd    = np.load(data_dir / "msd.npy")
boxL   = frames.max()          # 箱サイズ (スケール取得)

# --- UI ---
idx = st.slider("シミュレーションフレーム", 0, len(frames)-1, 0)
pos = frames[idx]
T   = temps[idx]
msd_val = msd[idx]

# --- 3D 散布図 ---
fig = go.Figure(
    data=[go.Scatter3d(
        x=pos[:,0], y=pos[:,1], z=pos[:,2],
        mode='markers',
        marker=dict(size=4, color=pos[:,2], colorscale='Viridis')
    )]
)
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[0, boxL], visible=False),
        yaxis=dict(range=[0, boxL], visible=False),
        zaxis=dict(range=[0, boxL], visible=False),
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
)

left, right = st.columns([4,1])
with left:
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.metric("温度 (LJ 単位)", f"{T:.2f}")
    st.metric("平均二乗変位", f"{msd_val:.2f}")

st.markdown(
"""
**遊び方**  
- スライダーを左から右へ動かしてみてください。  
- 低温では原子がきれいな格子を保っていますが、温度が上がると振幅が大きくなり、ある温度（融解点）付近で一気に **“液体のように” バラバラ** になります。  
- 右側の「平均二乗変位」が急上昇する箇所が **融解温度** の目安です。
"""
)
