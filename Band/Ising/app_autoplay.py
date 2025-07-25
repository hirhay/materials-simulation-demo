import streamlit as st
import pandas as pd
from PIL import Image
import plotly.graph_objs as go
import os, time

st.set_page_config(
    page_title="キュリー温度体験＋M–T 曲線",
    layout="wide", page_icon="🧲"
)

# ---------------- 設定 ----------------
mats = [("Fe", "鉄（1043 K）", "red"),
        ("Ni", "ニッケル（627 K）", "green"),
        ("Gd", "ガドリニウム（293 K）", "blue")]

T_min, T_max, dT = 0, 1200, 5
temps = list(range(T_min, T_max + dT, dT))

# --- データ読み込み（キャッシュあり） ---
@st.cache_data
def load_all_data():
    """全材料の磁化データを一度に読み込み、キャッシュする"""
    data = {}
    for name, _, _ in mats:
        path = os.path.join("materials", name, "magnetization.csv")
        df = pd.read_csv(path)
        data[name] = df
    return data

all_dfs = load_all_data()

# --- セッションステート初期化 ---
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "play" not in st.session_state:
    st.session_state.play = False

# --- UI（左 = ボタン/スライダー） ---
with st.sidebar:
    st.title("操作パネル")
    # Play / Pause
    if st.button("▶️ Play" if not st.session_state.play else "⏸️ Pause"):
        st.session_state.play = not st.session_state.play
    # 手動スライダー
    idx = st.slider("温度インデックス", 0, len(temps) - 1,
                    st.session_state.idx, key="slider")
    st.session_state.idx = idx
    # 再生速度
    speed = st.select_slider("再生スピード (frames/sec)",
                             options=[1, 2, 5, 10], value=2)

# --- 描画用レイアウト ---
upper, lower = st.columns([3, 2])

# ---------------- 上段：スピン配置 ---------------
with upper:
    cols = st.columns(len(mats))
    for (name, label, _), col in zip(mats, cols):
        T = temps[st.session_state.idx]
        img_path = os.path.join("materials", name, f"{T:04}.png")
        img = Image.open(img_path)
        col.image(img, caption=f"{label}\nT = {T} K", use_container_width=True)

# ---------------- 下段：M–T 曲線 ----------------
with lower:
    fig = go.Figure()
    for name, label, color in mats:
        df = all_dfs[name]
        # 折れ線
        fig.add_trace(go.Scatter(
            x=df["T_K"], y=df["M_abs"],
            mode="lines", name=label, line=dict(color=color)))
        # 現在温度位置
        # x座標もDataFrameから取得するように修正
        current_T = df["T_K"][st.session_state.idx]
        current_M = df["M_abs"][st.session_state.idx]
        fig.add_trace(go.Scatter(
            x=[current_T],
            y=[current_M],
            mode="markers", marker=dict(size=10, color=color),
            showlegend=False))
    fig.update_layout(
        xaxis_title="温度 T [K]",
        yaxis_title="絶対磁化 │M│",
        height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- 自動再生ロジック ---------------
if st.session_state.play:
    time.sleep(1.0 / speed)                     # 再描画間隔
    st.session_state.idx = (st.session_state.idx + 1) % len(temps)
    st.rerun()