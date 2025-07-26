# app.py
def main():
    import streamlit as st
    import numpy as np
    import plotly.graph_objs as go
    from pathlib import Path
    import time

    st.set_page_config(page_title="原子めがねで見る融解", layout="wide")

    # --- フォントとスタイルのカスタムCSS ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
        html, body, [class*="st-"], [class*="css-"] {
            font-family: 'Noto Sans JP', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- 定数 ---
    LJ_TO_KELVIN_FACTOR = 120.0

    # --- データ読み込み ---
    @st.cache_data
    def load_data():
        data_dir = Path(__file__).parent / "data"
        required_files = ["frames.npy", "temps.npy", "msd.npy", "rdfs.npy", "rdf_r_axis.npy"]
        if not all((data_dir / f).exists() for f in required_files):
            st.error("データファイルが不足しています。`python precompute_melting.py` を実行して、計算を完了させてください。")
            st.stop()

        frames = np.load(data_dir / "frames.npy")
        temps = np.load(data_dir / "temps.npy")
        msd = np.load(data_dir / "msd.npy")
        rdfs = np.load(data_dir / "rdfs.npy")
        rdf_r = np.load(data_dir / "rdf_r_axis.npy")
        boxL = frames.max()
        return frames, temps, msd, rdfs, rdf_r, boxL

    frames, temps, msd, rdfs, rdf_r, boxL = load_data()
    n_frames = len(frames)

    # --- セッション状態の初期化 ---
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    if "is_autoplay" not in st.session_state:
        st.session_state.is_autoplay = True

    # --- コールバック関数の定義 ---
    def toggle_autoplay():
        st.session_state.is_autoplay = st.session_state.autoplay_widget

    def slider_on_change():
        st.session_state.is_autoplay = False
        st.session_state.frame_idx = st.session_state.slider_widget

    # --- ロジックの実行 (ウィジェット描画前) ---
    if st.session_state.is_autoplay:
        st.session_state.frame_idx = (st.session_state.frame_idx + 1) % n_frames

    # --- サイドバー (UI) ---
    st.sidebar.title("操作パネル")

    st.sidebar.checkbox(
        "自動再生", 
        key="autoplay_widget", 
        value=st.session_state.is_autoplay,
        on_change=toggle_autoplay
    )

    st.sidebar.slider(
        "シミュレーションフレーム", 0, n_frames - 1, 
        key='slider_widget',
        value=st.session_state.frame_idx,
        on_change=slider_on_change
    )

    animation_speed = st.sidebar.select_slider(
        "再生速度", options=['遅い', '普通', '速い', '最速'], value='遅い'
    )
    speed_map = {'遅い': 1.2, '普通': 0.4, '速い': 0.2, '最速': 0.06}

    # --- メイン表示 ---
    st.title("原子めがねで見る融解")

    # --- 描画処理 ---
    idx = st.session_state.frame_idx
    T_k = temps[idx] * LJ_TO_KELVIN_FACTOR
    msd_val = msd[idx]

    # --- レイアウト ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("3D原子配置")
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=frames[idx,:,0], y=frames[idx,:,1], z=frames[idx,:,2],
            mode='markers', marker=dict(size=5, color=frames[idx,:,2], colorscale='Viridis', opacity=0.8)
        )])
        fig_3d.update_layout(
            scene=dict(xaxis=dict(range=[0, boxL], visible=False), yaxis=dict(range=[0, boxL], visible=False), zaxis=dict(range=[0, boxL], visible=False), aspectmode='cube'),
            margin=dict(l=0, r=0, t=0, b=0), height=450 # 高さを調整
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")

        phys_col1, phys_col2 = st.columns(2)
        with phys_col1:
            st.metric("温度 (K)", f"{T_k:.1f}")
        with phys_col2:
            st.metric("平均二乗変位", f"{msd_val:.2f}")

    with col2:
        st.subheader("時間変化と構造")
        fig_msd = go.Figure()
        fig_msd.add_trace(go.Scatter(x=np.arange(n_frames), y=msd, mode='lines', name='MSD'))
        fig_msd.add_vline(x=idx, line_width=2, line_dash="dash", line_color="red")
        fig_msd.update_layout(title="平均二乗変位 (MSD)", xaxis_title="フレーム", yaxis_title="MSD", height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_msd, use_container_width=True)

        fig_rdf = go.Figure()
        fig_rdf.add_trace(go.Scatter(x=rdf_r, y=rdfs[idx], mode='lines', name='g(r)'))
        fig_rdf.update_layout(title="動径分布関数 g(r)", xaxis_title="距離 r (LJ単位)", yaxis_title="g(r)", yaxis=dict(range=[0, 4]), height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_rdf, use_container_width=True)

    # --- ヘルプ/説明セクション ---
    st.markdown("---")
    with st.expander("遊び方と解説を見る"):
        st.markdown("""
        ### 遊び方
        1.  サイドバーの **「自動再生」** にチェックを入れると、融解の様子がアニメーションで再生されます。
        2.  **スライダー** を手で動かすと、自動再生が止まり、マニュアルで温度を変化させられます。
        3.  原子の動きと右側の2つのグラフが連動している様子を観察してみてください。

        ### 現象の解説
        -   **3D原子配置**: 低温では原子がきれいな格子（結晶）を組んでいますが、温��が上がると激しく振動し、ある温度（融解点）で一気に **“液体のように” バラバラ** になります。原子の色は高さ(Z軸)に対応しています。
        -   **平均二乗変位 (MSD)**: 原子が初期位置からどれだけ動いたかを示します。固体では小さな値で安定しますが、液体になると原子が自由に動き回るため、時間と共に **急激に増加** します。
        -   **動径分布関数 (g(r))**: ある原子から見た他の原子の分布です。**固体**では結晶構造を反映した **鋭いピーク** が見られ、**液体**になるとそのピークが **ブロードになり1に収束** していきます。このシミュレーションでは、**約190 K** で融解が起きています（実際のアルゴンの融点は約84 Kです）。
        """)

    # --- 自動再生のための再実行 ---
    if st.session_state.is_autoplay:
        time.sleep(speed_map[animation_speed])
        st.rerun()

if __name__ == '__main__':
    main()