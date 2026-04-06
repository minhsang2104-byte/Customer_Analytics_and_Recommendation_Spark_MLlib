# =============================================================================
# [13] STREAMLIT APP - HOÀN CHỈNH 6 TRANG
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os
from datetime import datetime

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 1em;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "./streamlit_data"

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(filename):
    path = f"{DATA_PATH}/{filename}"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_model(filename):
    path = f"{DATA_PATH}/{filename}"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Python.svg/48px-Python.svg.png", width=48)
    st.title("🛒 Olist Analytics")
    st.markdown("---")

    page = st.radio("📌 Chọn trang:", [
        "📊 Dashboard",
        "👥 Phân khúc KH",
        "🎯 Khuyến nghị",
        "📈 Xu hướng",
        "🔮 Dự đoán",
        "⚙️ Admin"
    ])

    st.markdown("---")
    st.info("**Nhóm 6 - Big Data Analytics**\n\nGVHD: Hồ Nhựt Minh")
    st.markdown(f"*Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}*")

# =============================================================================
# TRANG 1: DASHBOARD
# =============================================================================
if page == "📊 Dashboard":
    st.title("📊 Dashboard Tổng Quan")

    stats = load_csv("stats.csv")

    if not stats.empty:
        s = stats.iloc[0]

        # KPI Cards
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(s['total_orders']):,}</div>
                <div class="metric-label">🧾 Tổng Đơn</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(s['total_customers']):,}</div>
                <div class="metric-label">👥 Khách Hàng</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(s['total_products']):,}</div>
                <div class="metric-label">📦 Sản Phẩm</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">R$ {s['avg_payment']:.2f}</div>
                <div class="metric-label">💰 Giá TB</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{s['avg_review']:.2f}/5</div>
                <div class="metric-label">⭐ Review TB</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Biểu đồ
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⭐ Phân phối Review Score")
        df_rev = load_csv("review_dist.csv")
        if not df_rev.empty:
            fig = px.bar(df_rev, x="review_score", y="count",
                        color="count", color_continuous_scale="Blues")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💳 Phương thức thanh toán")
        df_pay = load_csv("payment_dist.csv")
        if not df_pay.empty:
            fig = px.pie(df_pay, values="count", names="payment_type",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Top categories
    st.subheader("🏆 Top 10 Danh Mục Sản Phẩm")
    df_cat = load_csv("top_categories.csv")
    if not df_cat.empty:
        fig = px.bar(df_cat, x="count", y="product_category_name_english",
                    orientation="h", color="count",
                    color_continuous_scale="Viridis")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Classification metrics
    st.subheader("📋 Kết quả mô hình Classification")
    df_cls = load_csv("cls_metrics.csv")
    if not df_cls.empty:
        st.dataframe(df_cls.style.highlight_max(
            subset=["Accuracy","F1","AUC"], color="#d4edda"
        ), use_container_width=True)

# =============================================================================
# TRANG 2: PHÂN KHÚC KHÁCH HÀNG
# =============================================================================
elif page == "👥 Phân khúc KH":
    st.title("👥 Phân Khúc Khách Hàng (RFM + KMeans)")

    df_rfm = load_csv("rfm_clusters.csv")

    if not df_rfm.empty:
        # ✅ FIX: KHÔNG dùng emoji trong data, chỉ hiển thị ở UI
        CLUSTER_LABELS = {
            0: "Khach Moi",
            1: "It Hoat Dong",
            2: "Trung Binh",
            3: "Than Thiet",
            4: "VIP"
        }

        # ✅ FIX: Map emoji ở display layer, không lưu vào CSV
        CLUSTER_EMOJI = {
            "Khach Moi": "🆕",
            "It Hoat Dong": "📉",
            "Trung Binh": "💛",
            "Than Thiet": "⭐",
            "VIP": "👑"
        }

        if "prediction" in df_rfm.columns:
            df_rfm["Nhom"] = df_rfm["prediction"].map(CLUSTER_LABELS).fillna("Khac")
            # ✅ FIX: Thêm emoji vào cột hiển thị riêng, không ghi vào CSV
            df_rfm["Nhóm_Display"] = df_rfm["Nhom"].apply(
                lambda x: f"{CLUSTER_EMOJI.get(x, '')} {x}"
            )

        tab1, tab2, tab3 = st.tabs(["📊 Tổng quan", "🔍 Chi tiết", "📥 Upload mới"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Tỷ lệ từng nhóm")
                # ✅ FIX: Dùng cột Nhom (không emoji) cho pie chart
                fig = px.pie(df_rfm, names="Nhom",
                            color="Nhom",
                            color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Monetary theo nhóm")
                # ✅ FIX: Dùng cột Nhom cho box plot
                fig = px.box(df_rfm, x="Nhom", y="monetary",
                            color="Nhom", title="Phân bố Monetary")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # RFM Scatter
            st.subheader("RFM Scatter Plot")
            sample_size = min(3000, len(df_rfm))
            df_sample = df_rfm.sample(sample_size, random_state=42)

            fig = px.scatter(df_sample, x="recency", y="monetary",
                           size="frequency", color="Nhom",
                           title="RFM Analysis (Recency vs Monetary)",
                           opacity=0.6, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Thống kê chi tiết từng nhóm")

            cluster_stats = df_rfm.groupby("Nhom").agg(
                Số_KH=("recency","count"),
                Recency_TB=("recency","mean"),
                Frequency_TB=("frequency","mean"),
                Monetary_TB=("monetary","mean")
            ).round(2).reset_index()

            # ✅ FIX: Thêm emoji vào display table
            cluster_stats["Nhóm"] = cluster_stats["Nhom"].apply(
                lambda x: f"{CLUSTER_EMOJI.get(x, '')} {x}"
            )

            st.dataframe(cluster_stats[["Nhóm", "Số_KH", "Recency_TB", "Frequency_TB", "Monetary_TB"]],
                        use_container_width=True)

            # Strategy recommendations
            st.subheader("💡 Chiến lược đề xuất")

            strategies = {
                "Khach Moi": "Upsell, Cross-sell, Welcome campaign",
                "It Hoat Dong": "Re-engagement, Special offers",
                "Trung Binh": "Loyalty program, Personalization",
                "Than Thiet": "Retention, Referral program",
                "VIP": "VIP treatment, Exclusive offers"
            }

            for cluster, strategy in strategies.items():
                emoji = CLUSTER_EMOJI.get(cluster, "")
                st.write(f"**{emoji} {cluster}**: {strategy}")

        with tab3:
            st.subheader("Upload dữ liệu khách hàng mới")
            uploaded = st.file_uploader("Chọn file CSV", type="csv")

            if uploaded:
                df_new = pd.read_csv(uploaded)
                st.write(f"✅ Đã upload: {len(df_new):,} khách hàng")
                st.dataframe(df_new.head(), use_container_width=True)

                if st.button("Phân tích clusters"):
                    st.success("Đã phân tích thành công! (Demo)")
# =============================================================================
# TRANG 3: KHUYẾN NGHỊ (ALS)
# =============================================================================
elif page == "🎯 Khuyến nghị":
    st.title("🎯 Hệ Thống Khuyến Nghị (ALS)")
    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <strong>📊 Thông tin mô hình ALS:</strong><br>
        • Phương pháp: Collaborative Filtering<br>
        • Rank: 10 | MaxIter: 10 | RegParam: 0.01<br>
        • Đánh giá dựa trên review_score của khách hàng
    </div>
    """, unsafe_allow_html=True)

    # Load ALS recommendations
    df_als = load_csv("als_recommendations.csv")

    if not df_als.empty:
        st.subheader("🔍 Tra cứu khuyến nghị")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Get max user_idx from data
            max_user = int(df_als["user_idx"].max())
            user_idx = st.number_input(
                "Nhập User Index:",
                min_value=0,
                max_value=max_user,
                value=0,
                step=1,
                help=f"Chọn user index từ 0 đến {max_user}"
            )

        with col2:
            search_btn = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)

        # Initialize session state for user_idx
        if 'current_user_idx' not in st.session_state:
            st.session_state.current_user_idx = 0

        if search_btn or user_idx != st.session_state.current_user_idx:
            st.session_state.current_user_idx = user_idx

            # ✅ FIX: Lọc đúng user và lấy top 10
            user_recs = df_als[df_als["user_idx"] == user_idx].copy()

            if not user_recs.empty:
                # Sort by rating descending và lấy top 10
                top_recs = user_recs.sort_values("rating", ascending=False).head(10)

                st.success(f"✅ Tìm thấy {len(top_recs)} sản phẩm khuyến nghị cho User #{user_idx}")

                # Hiển thị table
                st.subheader("📋 Top 10 Sản Phẩm Khuyến Nghị")

                # Format table cho đẹp
                display_df = top_recs[['item_idx', 'rating']].copy()
                display_df.columns = ['📦 Product Index', '⭐ Predicted Rating']
                display_df['⭐ Predicted Rating'] = display_df['⭐ Predicted Rating'].round(3)

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ✅ FIX: Bar chart cho top 10
                st.subheader("📊 Biểu Đồ Rating")

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=top_recs['item_idx'].astype(str),
                    y=top_recs['rating'],
                    marker_color='rgba(50, 171, 96, 0.7)',
                    marker_line_color='rgb(50, 171, 96)',
                    marker_line_width=1.5,
                    text=top_recs['rating'].round(2),
                    textposition='auto',
                    textfont=dict(size=10, color='rgb(49,130,40)')
                ))

                fig.update_layout(
                    title=f"🎯 Top 10 Sản Phẩm Khuyến Nghị - User #{user_idx}",
                    xaxis_title="📦 Product Index",
                    yaxis_title="⭐ Predicted Rating",
                    xaxis=dict(tickangle=-45),
                    yaxis=dict(range=[0, max(top_recs['rating']) * 1.2]),
                    height=500,
                    showlegend=False,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Số sản phẩm", f"{len(top_recs)}")
                with col2:
                    st.metric("Rating cao nhất", f"{top_recs['rating'].max():.3f}")
                with col3:
                    st.metric("Rating trung bình", f"{top_recs['rating'].mean():.3f}")

            else:
                st.warning(f"⚠️ Không tìm thấy khuyến nghị cho User #{user_idx}")
                st.info("💡 Thử với user index khác (0-100) để xem kết quả")

        st.divider()

        # Preview data
        with st.expander("📋 Xem mẫu dữ liệu ALS (50 dòng đầu)", expanded=False):
            st.dataframe(df_als.head(50), use_container_width=True)

            st.markdown("""
            **Giải thích các cột:**
            - `user_idx`: Index của khách hàng (đã được StringIndexer convert)
            - `item_idx`: Index của sản phẩm (đã được StringIndexer convert)
            - `rating`: Điểm đánh giá dự đoán (dựa trên review_score)
            """)

    else:
        st.error("❌ Không tìm thấy dữ liệu khuyến nghị!")
        st.info("💡 Vui lòng chạy cell Export trong notebook để tạo file als_recommendations.csv")
# =============================================================================
# TRANG 4: XU HƯỚNG
# =============================================================================
elif page == "📈 Xu hướng":
    st.title("📈 Xu Hướng Mua Sắm (FP-Growth)")

    df_rules = load_csv("association_rules.csv")

    if not df_rules.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng rules", len(df_rules))
        col2.metric("Confidence TB", f"{df_rules['confidence'].mean():.3f}")
        col3.metric("Lift TB", f"{df_rules['lift'].mean():.3f}")

        min_conf = st.slider("Confidence tối thiểu", 0.0, 1.0, 0.1)
        min_lift = st.slider("Lift tối thiểu", 1.0, 10.0, 1.0)

        df_filtered = df_rules[
            (df_rules["confidence"] >= min_conf) &
            (df_rules["lift"] >= min_lift)
        ].sort_values("lift", ascending=False)

        st.dataframe(df_filtered.head(20), use_container_width=True)

# =============================================================================
# TRANG 5: DỰ ĐOÁN (FIXED + IMPROVED)
# =============================================================================
elif page == "🔮 Dự đoán":
    st.title("🔮 Dự Đoán")

    tab1, tab2 = st.tabs(["⭐ Review Score", "💰 Giá Trị Đơn"])

    # Load models
    gbt_model = load_model("gbt_model.pkl")
    rfr_model = load_model("rf_regressor.pkl")

    # ── TAB 1: REVIEW SCORE PREDICTION ──────────────────────────────────────
    with tab1:
        st.subheader("⭐ Dự Đoán Đánh Giá Khách Hàng")

        # Info box với mẹo để được "HÀI LÒNG"
        st.info("""
        **💡 Mẹo để đạt được sự hài lòng:**
        - Giao hàng **sớm hơn dự kiến** (độ trễ âm: -5 đến 0 ngày)
        - Phí vận chuyển hợp lý (**< 20 R$**)
        - Giá sản phẩm vừa phải (**30-80 R$**)
        - Thanh toán bằng **thẻ tín dụng**
        """)

        col1, col2 = st.columns(2)

        with col1:
            price = st.number_input(
                "💵 Giá sản phẩm (R$)",
                min_value=0.0,
                max_value=10000.0,
                value=50.0,
                step=1.0,
                help="Giá bán của sản phẩm"
            )
            freight = st.number_input(
                "📦 Phí vận chuyển (R$)",
                min_value=0.0,
                max_value=500.0,
                value=15.0,
                step=1.0,
                help="Chi phí giao hàng"
            )

        with col2:
            delay = st.number_input(
                "⏱️ Độ trễ giao hàng (ngày)",
                min_value=-30,
                max_value=100,
                value=0,
                step=1,
                help="Số ngày chênh lệch so với dự kiến (âm = giao sớm, dương = giao trễ)"
            )
            # ✅ FIX: Hiển thị tên phương thức thanh toán thay vì số
            ptype = st.selectbox(
                "💳 Phương thức thanh toán",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "💳 Thẻ tín dụng (credit_card)",
                    1: "📄 Hóa đơn (boleto)",
                    2: "🎫 Phiếu mua hàng (voucher)",
                    3: "💳 Thẻ ghi nợ (debit_card)"
                }[x],
                help="Chọn phương thức thanh toán"
            )

        # Prediction button
        if st.button("🔮 Dự Đoán Review Score", type="primary", use_container_width=True):
            if gbt_model:
                # Prepare input
                X = np.array([[price, freight, delay, ptype]])
                pred = gbt_model.predict(X)[0]
                proba = gbt_model.predict_proba(X)[0]

                # ✅ FIX: Hiển thị kết quả với confidence
                if pred == 1:
                    st.success(f"**✅ Kết quả: HÀI LÒNG (4-5 sao)**", icon="✅")
                    st.balloons()
                else:
                    st.error(f"**❌ Kết quả: KHÔNG HÀI LÒNG (1-3 sao)**", icon="❌")

                # ✅ FIX: Hiển thị xác suất (confidence)
                st.markdown("### 📊 Xác suất dự đoán:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Không hài lòng",
                        f"{proba[0]*100:.1f}%",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Hài lòng",
                        f"{proba[1]*100:.1f}%",
                        delta=None
                    )

                # ✅ FIX: Bar chart visualization
                fig = go.Figure(go.Bar(
                    x=["Không hài lòng", "Hài lòng"],
                    y=[proba[0], proba[1]],
                    marker_color=["#e74c3c", "#2ecc71"],
                    text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="📊 Phân bố Xác suất",
                    yaxis_title="Probability",
                    yaxis_tickformat=".0%",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # ✅ FIX: Gợi ý cải thiện nếu không hài lòng
                if pred == 0:
                    st.markdown("---")
                    st.subheader("🎯 Gợi ý Cải thiện")
                    st.info("""
                    **Để đạt được sự hài lòng, cân nhắc:**
                    - ⏱️ Giảm thời gian giao hàng (giao sớm hơn)
                    - 📦 Giảm phí vận chuyển
                    - 💵 Điều chỉnh giá sản phẩm hợp lý
                    - 💳 Khuyến khích thanh toán bằng thẻ tín dụng
                    """)
            else:
                st.error("❌ Model chưa được load. Kiểm tra file gbt_model.pkl")

    # ── TAB 2: ORDER VALUE PREDICTION ───────────────────────────────────────
    with tab2:
        st.subheader("💰 Dự Đoán Giá Trị Đơn Hàng")

        # Info box
        st.info("""
        **📊 Thông tin mô hình:**
        - Model: Random Forest Regressor
        - RMSE: 166.03 BRL
        - R²: 0.6586 (65.86% phương sai được giải thích)
        """)

        # ✅ FIX: Inputs riêng cho Tab 2 (không dùng chung với Tab 1)
        col1, col2 = st.columns(2)

        with col1:
            price2 = st.number_input(
                "💵 Giá sản phẩm (R$)",
                min_value=0.0,
                max_value=10000.0,
                value=80.0,
                step=1.0,
                key="p2"  # ✅ FIX: Unique key
            )
            freight2 = st.number_input(
                "📦 Phí vận chuyển (R$)",
                min_value=0.0,
                max_value=500.0,
                value=20.0,
                step=1.0,
                key="f2"  # ✅ FIX: Unique key
            )

        with col2:
            delay2 = st.number_input(
                "⏱️ Độ trễ (ngày)",
                min_value=-30,
                max_value=100,
                value=0,
                step=1,
                key="d2"  # ✅ FIX: Unique key
            )
            ptype2 = st.selectbox(
                "💳 Phương thức thanh toán",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "💳 Thẻ tín dụng (credit_card)",
                    1: "📄 Hóa đơn (boleto)",
                    2: "🎫 Phiếu mua hàng (voucher)",
                    3: "💳 Thẻ ghi nợ (debit_card)"
                }[x],
                key="pt2"  # ✅ FIX: Unique key
            )

        # ✅ FIX: Button riêng cho Tab 2
        if st.button("💰 Dự Đoán Giá Trị", type="primary", use_container_width=True, key="btn_reg"):
            if rfr_model:
                X = np.array([[price2, freight2, delay2, ptype2]])
                pred_val = rfr_model.predict(X)[0]

                # Display result
                st.success(f"**💰 Giá trị đơn hàng dự đoán: R$ {pred_val:,.2f}**")

                # ✅ FIX: Gauge chart visualization
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_val,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Giá trị đơn hàng (R$)", 'font': {'size': 24}},
                    delta={'reference': 200, 'increasing': {'color': "RebeccaPurple"}},
                    gauge={
                        'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 100], 'color': '#ffebee'},
                            {'range': [100, 300], 'color': '#fff3e0'},
                            {'range': [300, 500], 'color': '#e8f5e9'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 400
                        }
                    }
                ))
                st.plotly_chart(gauge, use_container_width=True)

                # ✅ FIX: Interpretation
                col1, col2, col3 = st.columns(3)
                with col1:
                    if pred_val < 100:
                        st.info("📊 **Đơn hàng nhỏ**")
                    elif pred_val < 300:
                        st.success("📊 **Đơn hàng trung bình**")
                    else:
                        st.warning("📊 **Đơn hàng lớn**")
            else:
                st.error("❌ Model chưa được load. Kiểm tra file rf_regressor.pkl")

# =============================================================================
# TRANG 6: ADMIN
# =============================================================================
elif page == "⚙️ Admin":
    st.title("⚙️ Admin Panel")

    tab1, tab2, tab3 = st.tabs(["📦 Trạng thái", "📊 Báo cáo", "🔄 Retrain"])

    # ── TAB 1: TRẠNG THÁI FILES ──────────────────────────────────────────────
    with tab1:
        st.subheader("Trạng thái Files")

        files = [
            "stats.csv", "rfm_clusters.csv", "als_recommendations.csv",
            "association_rules.csv", "cls_metrics.csv", "reg_metrics.csv",
            "gbt_model.pkl", "rf_regressor.pkl"
        ]

        for f in files:
            exists = os.path.exists(f"{DATA_PATH}/{f}")
            size = os.path.getsize(f"{DATA_PATH}/{f}") // 1024 if exists else 0
            status = "✅" if exists else "❌"
            size_str = f"{size} KB" if exists else "— Chưa có"
            st.write(f"{status} `{f}` — {size_str}")

        st.divider()

        # Lịch sử retrain
        st.subheader("📜 Lịch Sử Retrain")

        if 'retrain_history' not in st.session_state:
            st.session_state['retrain_history'] = []

        if st.session_state['retrain_history']:
            for record in st.session_state['retrain_history'][-5:]:
                st.write(f"🕐 {record['time']} - {record['status']}")
        else:
            st.info("Chưa có lịch sử retrain")

    # ── TAB 2: BÁO CÁO ĐÁNH GIÁ ──────────────────────────────────────────────
    with tab2:
        st.subheader("Báo cáo Classification")

        df_cls = load_csv("cls_metrics.csv")
        if not df_cls.empty:
            st.dataframe(df_cls, use_container_width=True)

            fig = px.bar(df_cls, x="Model", y=["Accuracy","F1","AUC"],
                        barmode="group", title="So sánh các mô hình Classification",
                        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Chưa có dữ liệu classification metrics")

        st.divider()

        st.subheader("Báo cáo Regression")

        df_reg = load_csv("reg_metrics.csv")
        if not df_reg.empty:
            st.dataframe(df_reg, use_container_width=True)

            fig = px.bar(df_reg, x="Model", y=["RMSE","MAE"],
                        barmode="group", title="So sánh các mô hình Regression",
                        color_discrete_sequence=["#e74c3c", "#3498db"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Chưa có dữ liệu regression metrics")

    # ── TAB 3: UPLOAD & RETRAIN ──────────────────────────────────────────────
    with tab3:
        st.subheader("📤 Upload Dữ Liệu Mới")

        uploaded = st.file_uploader("Chọn file CSV", type="csv")

        if uploaded:
            df_new = pd.read_csv(uploaded)
            st.write(f"✅ Đã upload: {len(df_new):,} dòng, {len(df_new.columns)} cột")
            st.dataframe(df_new.head(), use_container_width=True)

            # Lưu file
            save_path = f"{DATA_PATH}/new_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_new.to_csv(save_path, index=False)
            st.success(f"✅ Đã lưu: {save_path}")

        st.divider()

        st.subheader("🔄 Retrain Models")
        st.info("""
        **💡 Lưu ý quan trọng:**
        - Retrain thực sự cần chạy notebook Colab (30-60 phút)
        - Nút bên dưới là demo chức năng UI
        - Sau khi retrain thực sự, refresh trang để xem kết quả mới
        """)

        if st.button("🔄 Retrain Models", type="primary", use_container_width=True):
            with st.spinner("⏳ Đang retrain models..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("🔄 Đang xử lý dữ liệu...")
                    elif i < 60:
                        status_text.text("🔧 Đang train Classification models...")
                    elif i < 80:
                        status_text.text("🔧 Đang train Regression models...")
                    else:
                        status_text.text("💾 Đang lưu models...")
                    time.sleep(0.2)

                status_text.text("✅ Hoàn thành!")

            # Lưu vào session state
            retrain_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            st.session_state['retrain_history'].append({
                'time': retrain_time,
                'status': '✅ Thành công'
            })

            st.success(f"✅ Retrain hoàn thành lúc {retrain_time}!")

            st.markdown("""
            **📋 Các bước tiếp theo:**
            1. Kiểm tra tab "📊 Báo cáo" để xem metrics
            2. Test lại trang "🔮 Dự đoán"
            3. Download models từ tab Trạng thái
            """)

        st.divider()

        # Download buttons
        st.subheader("📥 Export Models & Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            gbt_path = f"{DATA_PATH}/gbt_model.pkl"
            if os.path.exists(gbt_path):
                with open(gbt_path, "rb") as f:
                    st.download_button(
                        label="📥 GBT Model",
                        data=f.read(),
                        file_name="gbt_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )

        with col2:
            rfr_path = f"{DATA_PATH}/rf_regressor.pkl"
            if os.path.exists(rfr_path):
                with open(rfr_path, "rb") as f:
                    st.download_button(
                        label="📥 RF Model",
                        data=f.read(),
                        file_name="rf_regressor.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )

        with col3:
            metrics_path = f"{DATA_PATH}/cls_metrics.csv"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    st.download_button(
                        label="📥 Metrics",
                        data=f.read(),
                        file_name="cls_metrics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray'>
    <strong>HCMUTE | Nhóm 6 - Big Data Analytics</strong><br>
    GVHD: Hồ Nhựt Minh
</div>
""", unsafe_allow_html=True)