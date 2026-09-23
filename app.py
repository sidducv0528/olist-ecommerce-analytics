# -*- coding: utf-8 -*-
"""
app.py – Olist E-Commerce Analytics Dashboard
Author: Siddu Varikuppala | AICTE | IBM SkillsBuild 2026
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, os, joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #57606a; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #1f2328; }
    .metric-delta { font-size: 12px; color: #3b82d4; margin-top: 2px; }
    [data-testid="stMetric"] { background: #f7f8fa; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data …")
def load_data():
    # Load full delivered dataset — no sampling so KPIs match the report (97,007 orders)
    df = pd.read_csv('data/delivered.csv',
                     parse_dates=['order_purchase_timestamp'])
    return df

@st.cache_data(show_spinner="Loading RFM …")
def load_rfm():
    return pd.read_csv('data/rfm_segments.csv')

@st.cache_data(show_spinner="Loading metrics …")
def load_metrics():
    with open('data/metrics_fg.json') as f:
        return json.load(f)

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    return joblib.load('models/best_model.joblib')

# ── Safe loads ────────────────────────────────────────────────────────────────
try:
    df_raw   = load_data()
    rfm      = load_rfm()
    metrics  = load_metrics()
    model    = load_model()
    DATA_OK  = True
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")
st.sidebar.caption("Applied to Overview and Charts pages")

years = sorted(df_raw['purchase_year'].dropna().astype(int).unique())
sel_years = st.sidebar.multiselect("Year", years, default=years)

states = sorted(df_raw['customer_state'].dropna().unique())
sel_states = st.sidebar.multiselect("Customer State", states, default=states)

cats = sorted(df_raw['product_category_name_english'].dropna().unique())
sel_cats = st.sidebar.multiselect("Product Category", cats, default=cats)

@st.cache_data(show_spinner=False)
def filter_data(years_t, states_t, cats_t):
    mask = (
        df_raw['purchase_year'].isin(years_t) &
        df_raw['customer_state'].isin(states_t) &
        df_raw['product_category_name_english'].isin(cats_t)
    )
    return df_raw[mask].copy()

df = filter_data(tuple(sel_years), tuple(sel_states), tuple(sel_cats))
if len(df) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Navigation tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "📈 Charts", "👥 Customer Segments", "🤖 Prediction"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🛒 Olist E-Commerce Analytics")
    st.caption(f"Dataset: Brazilian E-Commerce Public Dataset by Olist | "
               f"Showing {len(df):,} of {len(df_raw):,} delivered orders")

    total_rev    = df['total_price'].sum()
    total_orders = len(df)
    avg_review   = df['review_score'].mean()
    late_pct     = df['is_late'].mean() * 100
    avg_del_days = df['delivery_days'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Revenue",    f"R${total_rev:,.0f}")
    c2.metric("📦 Total Orders",     f"{total_orders:,}")
    c3.metric("⭐ Avg Review Score", f"{avg_review:.2f} / 5")
    c4.metric("🚚 Late Deliveries",  f"{late_pct:.1f}%")
    c5.metric("⏱ Avg Delivery Days", f"{avg_del_days:.1f}d")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Review Score Distribution")
        rev_dist = df['review_score'].dropna().astype(int).value_counts().sort_index()
        fig = px.bar(x=rev_dist.index.astype(str), y=rev_dist.values,
                     labels={'x':'Review Score','y':'Orders'},
                     color=rev_dist.index.astype(str),
                     color_discrete_sequence=px.colors.diverging.RdYlGn,
                     title="Review Score Distribution")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Payment Types")
        pay = df['payment_type'].value_counts().reset_index()
        pay.columns = ['type','count']
        fig = px.pie(pay, values='count', names='type',
                     title="Payment Type Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Key Findings")
    st.info(
        f"**Late deliveries** (6.8% of orders) score **2.27** on average vs **4.29** "
        f"for on-time — a statistically large effect (Mann-Whitney r=0.64, p<0.001). "
        f"State **RR** averages **29 days** delivery vs the national mean of **12 days**."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📈 Exploratory Charts")

    # Revenue trend
    df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    monthly = (df.groupby('year_month')
               .agg(revenue=('total_price','sum'), orders=('order_id','count'))
               .reset_index())

    fig = go.Figure()
    fig.add_bar(x=monthly['year_month'], y=monthly['revenue'],
                name='Revenue (R$)', marker_color='#4C9BE8', opacity=0.7,
                yaxis='y1')
    fig.add_scatter(x=monthly['year_month'], y=monthly['orders'],
                    name='Orders', mode='lines+markers',
                    marker=dict(color='#E84C4C', size=5), yaxis='y2')
    fig.update_layout(
        title='Monthly Revenue and Order Volume',
        xaxis_title='Month',
        yaxis=dict(title='Revenue (R$)', tickformat=',.0f'),
        yaxis2=dict(title='Orders', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Top categories
        top_cat = (df.groupby('product_category_name_english')['total_price']
                   .sum().sort_values(ascending=False).head(10).reset_index())
        top_cat.columns = ['category','revenue']
        fig = px.bar(top_cat, x='revenue', y='category', orientation='h',
                     title='Top 10 Categories by Revenue',
                     labels={'revenue':'Revenue (R$)','category':'Category'},
                     color='revenue', color_continuous_scale='Blues')
        fig.update_layout(height=380, showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Delivery by state
        state_del = (df.groupby('customer_state')['delivery_days']
                     .mean().sort_values(ascending=False).reset_index())
        state_del.columns = ['state','avg_days']
        fig = px.bar(state_del, x='state', y='avg_days',
                     title='Avg Delivery Days by Customer State',
                     labels={'avg_days':'Avg Days','state':'State'},
                     color='avg_days', color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Late vs on-time by review
        cross = df.groupby(['is_late','review_score']).size().reset_index(name='count')
        cross['Late'] = cross['is_late'].map({0:'On-Time', 1:'Late'})
        fig = px.box(df.dropna(subset=['review_score']),
                     x='is_late', y='review_score',
                     labels={'is_late':'Late (1=yes)','review_score':'Review Score'},
                     title='Review Score: Late vs On-Time',
                     color='is_late',
                     color_discrete_map={0:'#7EC8A4', 1:'#E84C4C'})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Freight ratio distribution
        fig = px.histogram(df, x='freight_ratio', nbins=60,
                           title='Freight Ratio Distribution',
                           labels={'freight_ratio':'Freight / Price Ratio'},
                           color_discrete_sequence=['#4C9BE8'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CUSTOMER SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("👥 Customer Segments (RFM + KMeans)")

    best_k       = metrics.get('best_k', 3)
    best_sil     = metrics.get('best_silhouette', 0)
    num_cust     = metrics.get('num_customers', len(rfm))
    rfm_snapshot = metrics.get('rfm_snapshot', 'N/A')

    st.info(
        f"**{num_cust:,} customers** segmented into **{best_k} clusters** using KMeans "
        f"(silhouette={best_sil:.4f}). Snapshot date: {rfm_snapshot}."
    )

    seg_profile = (rfm.groupby('segment')
                   .agg(count=('customer_id','count'),
                        avg_recency=('recency','mean'),
                        avg_frequency=('frequency','mean'),
                        avg_monetary=('monetary','mean'),
                        avg_rfm_score=('RFM_score','mean'))
                   .round(2).reset_index())
    seg_profile.columns = ['Segment','Count','Avg Recency (d)',
                           'Avg Frequency','Avg Monetary (R$)','Avg RFM Score']
    st.dataframe(seg_profile.sort_values('Avg RFM Score', ascending=False),
                 use_container_width=True, hide_index=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        seg_dist = rfm['segment'].value_counts().reset_index()
        seg_dist.columns = ['segment','count']
        fig = px.bar(seg_dist.sort_values('count', ascending=True),
                     x='count', y='segment', orientation='h',
                     title='Customer Count per Segment',
                     color='segment',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_s2:
        fig = px.scatter(rfm, x='recency', y='monetary',
                         color='segment', size='monetary',
                         size_max=15,
                         title='Recency vs Monetary by Segment',
                         labels={'recency':'Recency (days)','monetary':'Monetary (R$)'},
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         opacity=0.5)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Interpretation")
    st.caption(
        "Note: 96% of Olist customers placed exactly one order (frequency = 1), "
        "so segmentation is driven by Recency and Monetary value only. "
        "Frequency does not differentiate these customers."
    )
    seg_desc = {
        'High-Value Buyers':  'Highest monetary spend (avg R$689). Recent purchase. Priority retention — exclusive offers, early access.',
        'Recent Buyers':      'Recently active, moderate spend (avg R$103). Nurture with follow-up campaigns to build loyalty.',
        'Lapsed Buyers':      'Long recency (~389 days), low recent activity. Win-back campaigns: personalised voucher or recommendation.',
    }
    for seg, desc in seg_desc.items():
        if seg in rfm['segment'].values:
            st.markdown(f"**{seg}:** {desc}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("🤖 Low-Review Risk Prediction")

    best_model_name = metrics.get('best_model_name', 'Random Forest')
    tuned_m         = metrics.get('tuned_metrics', {})

    st.info(
        f"Model: **Tuned {best_model_name}** | "
        f"F1={tuned_m.get('f1',0):.4f} | "
        f"ROC-AUC={tuned_m.get('roc_auc',0):.4f} | "
        f"Recall={tuned_m.get('recall',0):.4f}"
    )
    st.caption(
        "This is a **post-delivery model**: it uses delivery outcome features "
        "(actual delivery days, delay vs estimate, is_late) plus order and seller context "
        "to predict whether a delivered order will receive a review score ≤ 2. "
        "Performance is moderate (F1 0.46, AUC 0.77) — treat outputs as a risk flag, "
        "not a hard decision. No review text or score is used as input."
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Delivery**")
            delivery_days = st.number_input(
                "Delivery Days", min_value=0, max_value=200, value=12)
            delay_days = st.number_input(
                "Delay vs Estimate (days, negative=early)", min_value=-30, max_value=100, value=0)
            is_late = 1 if delay_days > 0 else 0
            st.caption(f"is_late = {is_late}")

        with c2:
            st.markdown("**Order**")
            total_price = st.number_input(
                "Total Price (R$)", min_value=0.0, max_value=5000.0, value=100.0, step=10.0)
            total_freight = st.number_input(
                "Total Freight (R$)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
            num_items = st.number_input(
                "Number of Items", min_value=1, max_value=20, value=1)
            payment_installments = st.number_input(
                "Payment Installments", min_value=1, max_value=24, value=1)

        with c3:
            st.markdown("**Context**")
            purchase_month   = st.slider("Purchase Month", 1, 12, 6)
            purchase_weekday = st.selectbox(
                "Weekday (0=Mon)", [0,1,2,3,4,5,6],
                format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
            purchase_hour    = st.slider("Purchase Hour", 0, 23, 14)

        st.markdown("**Product & Seller**")
        cols_b = st.columns(4)
        payment_type     = cols_b[0].selectbox(
            "Payment Type", ['credit_card','boleto','voucher','debit_card','not_defined'])
        product_category = cols_b[1].selectbox(
            "Product Category",
            sorted(df_raw['product_category_name_english'].dropna().unique()))
        customer_state   = cols_b[2].selectbox(
            "Customer State", sorted(df_raw['customer_state'].dropna().unique()))
        seller_state     = cols_b[3].selectbox(
            "Seller State", sorted(df_raw['seller_state'].dropna().unique()))

        submitted = st.form_submit_button("🔮 Predict Low-Review Risk", use_container_width=True)

    if submitted:
        freight_ratio = total_freight / total_price if total_price > 0 else 0.0
        input_row = pd.DataFrame([{
            'delivery_days':                  float(delivery_days),
            'estimated_vs_actual_delay_days': float(delay_days),
            'is_late':                        float(is_late),
            'total_price':                    float(total_price),
            'total_freight':                  float(total_freight),
            'freight_ratio':                  float(freight_ratio),
            'num_items':                      float(num_items),
            'payment_installments':           float(payment_installments),
            'purchase_month':                 float(purchase_month),
            'purchase_weekday':               float(purchase_weekday),
            'purchase_hour':                  float(purchase_hour),
            'payment_type':                   payment_type,
            'product_category_name_english':  product_category,
            'customer_state':                 customer_state,
            'seller_state':                   seller_state,
        }])

        try:
            pre   = model['preprocessor']
            clf   = model['classifier']
            X_in  = pre.transform(input_row)
            prob  = float(clf.predict_proba(X_in)[0][1])
            pred  = clf.predict(X_in)[0]

            st.divider()
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                if prob >= 0.5:
                    st.error(f"### ⚠️ HIGH RISK\n**Probability: {prob*100:.1f}%**")
                elif prob >= 0.3:
                    st.warning(f"### ⚡ MODERATE RISK\n**Probability: {prob*100:.1f}%**")
                else:
                    st.success(f"### ✅ LOW RISK\n**Probability: {prob*100:.1f}%**")

            with col_r2:
                st.markdown("**Input summary:**")
                st.write({
                    "Delivery days": delivery_days,
                    "Delay vs estimate": delay_days,
                    "Is late": bool(is_late),
                    "Freight ratio": f"{freight_ratio:.3f}",
                    "Category": product_category,
                    "State": customer_state,
                })

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Low-Review Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#E84C4C" if prob >= 0.5 else "#F5A623" if prob >= 0.3 else "#7EC8A4"},
                    'steps': [
                        {'range': [0,  30], 'color': "#d4edda"},
                        {'range': [30, 50], 'color': "#fff3cd"},
                        {'range': [50,100], 'color': "#f8d7da"},
                    ],
                    'threshold': {'line':{'color':'black','width':3},'thickness':0.75,'value': 50}
                }
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.divider()
    st.subheader("Model Details")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Model | {best_model_name} (Tuned) |
| Accuracy | {tuned_m.get('accuracy',0):.4f} |
| Precision | {tuned_m.get('precision',0):.4f} |
| Recall | {tuned_m.get('recall',0):.4f} |
| F1 Score | {tuned_m.get('f1',0):.4f} |
| ROC-AUC | {tuned_m.get('roc_auc',0):.4f} |
        """)
    with col_m2:
        st.markdown("""
**Why F1 & Recall matter more than Accuracy here:**
The dataset is imbalanced (~87% non-low). Predicting "not low" every time gives 87% accuracy
but catches zero real problems. We optimise for **F1** (precision-recall balance) and
**Recall** so operations teams can flag at-risk orders for intervention.

**Limitation:** Model performance is moderate (F1 0.46, AUC 0.77). It is best used
as a risk-prioritisation tool rather than a definitive classifier.
        """)
