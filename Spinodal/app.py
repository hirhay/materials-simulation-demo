# app.py
def main():
    import streamlit as st, numpy as np, plotly.express as px, time
    from pathlib import Path

    st.set_page_config(page_title="相分離シミュレーション：スピノーダル分解 vs 核形成・成長", layout="wide")
    st.title("相分離シミュレーション：スピノーダル分解 vs 核形成・成長")

    # --- データ読み込み ---
    @st.cache_data
    def load_data():
        data_dir = Path(__file__).parent / "data"
        conc_unstable = np.load(data_dir/"conc_unstable.npy")
        conc_nucleation = np.load(data_dir/"conc_nucleation.npy")
        times = np.load(data_dir/"time.npy")
        phys_params = np.load(data_dir/"phys_params.npy")
        return conc_unstable, conc_nucleation, times, phys_params

    try:
        conc_un, conc_nuc, times, phys_params = load_data()
        time_scale, L_unit = phys_params
        Nframe = len(times)
    except FileNotFoundError:
        st.error("データファイルが見つかりません。`precompute_spinodal.py` を先に実行してください。")
        st.stop()


    # --- セッション状態の初期化 ---
    if 'frame_idx' not in st.session_state:
        st.session_state.frame_idx = 0
    if 'autoplay' not in st.session_state:
        st.session_state.autoplay = True

    # --- サイドバー ---
    st.sidebar.title("コントロール")
    # チェックボックスをセッション状態に直接連付ける
    st.sidebar.checkbox("自動再生", key="autoplay")

    speed_options = {"遅い": 0.6, "普通": 0.2, "速い": 0.1, "超高速": 0.02}
    speed_key = st.sidebar.select_slider(
        "再生速度", options=speed_options.keys(), value="遅い"
    )
    sleep_duration = speed_options[speed_key]

    # --- 自動再生ロジック ---
    if st.session_state.autoplay:
        st.session_state.frame_idx = (st.session_state.frame_idx + 1) % Nframe

    # --- メインのスライダー ---
    # スライダーもセッション状態に直接関連付ける
    st.slider(
        "シミュレーションフレーム", 0, Nframe - 1,
        key="frame_idx"
    )

    # --- 現在のフレームデータを取得 ---
    idx = st.session_state.frame_idx
    c_un, c_nuc = conc_un[idx], conc_nuc[idx]
    t_sim = times[idx]
    t_phys = t_sim * time_scale  # 物理時間に換算

    # --- 画像表示 ---
    def show_img(arr, title):
        fig = px.imshow(arr, origin="lower", zmin=-0.8, zmax=0.8,
                        color_continuous_scale='viridis', aspect="equal")
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0), height=450,
            title=dict(text=title, font=dict(size=20))
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        show_img(c_un, "不安定系：スピノーダル分解")
    with right:
        show_img(c_nuc, "準安定系：核形成・成長")

    # --- 時間と説明文の表示 ---
    st.markdown(f"#### 無次元時間: {t_sim:.2f} | 物理時間: {t_phys*1e9:.1f} ナノ秒 (ns)")

    st.markdown("""
    - **左（スピノーダル分解）**: 成分が全体に混ざった不安定な状態から、濃度ゆらぎが成長して瞬時に細かい網目状の組織が形成され、時間とともにお互いが連結するように粗大化します。
    - **右（核形成・成長）**: エネルギー的に少しだけ安定な「準安定」状態から始まります。新しい相の「核」が一つ発生し、それを中心に相が成長していく様子がわかります。
    """)

    with st.expander("物理単位に関する補足"):
        st.markdown(f"""
        このシミュレーションでは、以下の物理パラメータ（一般的な合金の例）を用いて、無次元の計算時間を物理的な時間（秒）に換算しています。
        - **時間スケール**: `{time_scale:.2e} s`
        - **格子サイズ**: `{L_unit*1e9:.1f} nm`

        これらの値は、自由エネルギーの形状 `A`、界面エネルギー `κ`、原子の移動度 `M` によって決まります。
        """)

    # --- 自動再生の待機と再実行 ---
    if st.session_state.autoplay:
        time.sleep(sleep_duration)
        st.rerun()

if __name__ == '__main__':
    main()