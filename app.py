from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import analytics, prediction

st.set_page_config(
    page_title="DeliveryGuard | Delivery intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#4263EB"
NAVY = "#172554"
GREEN = "#16A34A"
RED = "#E5484D"
AMBER = "#D97706"
MUTED = "#64748B"
PALETTE = [BLUE, GREEN, AMBER, RED, "#8B5CF6", "#0891B2"]

st.markdown("""
<style>
:root { --ink:#172554; --muted:#475569; --line:#e5eaf3; --blue:#3155d9; }
html, body, [class*="css"] { font-family:Arial, sans-serif; }
.stApp { background:#F2F7FF; color:#172554; }

[data-testid="stAppViewContainer"] .stMarkdown,
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] .stCaption,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
    color:#172554;
}

.block-container {
    max-width:1500px;
    padding-top:1.7rem;
    padding-bottom:3rem;
}

h1, h2, h3 {
    font-family:Arial, sans-serif;
    color:#172554 !important;
    letter-spacing:-.025em;
}

[data-testid="stSidebar"] {
    background:#101b36;
    border-right:1px solid #263451;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color:#e2e8f0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color:#ffffff !important;
}

[data-testid="stSidebar"] hr {
    border-color:#475569;
}

[data-testid="stSidebar"] label {
    color:#e2e8f0 !important;
}

[data-testid="stMetric"] {
    background:#fff;
    border:1px solid var(--line);
    border-radius:16px;
    padding:1rem 1.1rem;
    box-shadow:0 4px 18px rgba(23,37,84,.035);
}

[data-testid="stMetricLabel"] {
    color:var(--muted);
    font-size:.82rem;
    font-weight:600;
}

[data-testid="stMetricValue"] {
    color:var(--ink);
    font-family:Arial,sans-serif;
    font-size:1.8rem;
    font-weight:800;
}

[data-testid="stMetricDelta"] {
    font-size:.76rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap:.35rem;
    background:#eef2f8;
    border-radius:14px;
    padding:.35rem;
}

button[data-baseweb="tab"],
[data-baseweb="tab-list"] button,
[data-testid="stTabs"] button[role="tab"],
.stTabs button[role="tab"] {
    color: #1e293b !important;
    opacity: 1 !important;
}

button[data-baseweb="tab"] *,
[data-baseweb="tab-list"] button *,
[data-testid="stTabs"] button[role="tab"] *,
.stTabs button[role="tab"] * {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    opacity: 1 !important;
    visibility: visible !important;
}

button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] *,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
    color: #1d4ed8 !important;
    -webkit-text-fill-color: #1d4ed8 !important;
    opacity: 1 !important;
}

button[data-baseweb="tab"]:hover,
button[data-baseweb="tab"]:hover *,
[data-testid="stTabs"] button[role="tab"]:hover,
[data-testid="stTabs"] button[role="tab"]:hover * {
    color: #1d4ed8 !important;
    -webkit-text-fill-color: #1d4ed8 !important;
}
.stButton button {
    border-radius:10px;
    font-weight:700;
}

.stSelectbox [data-baseweb="select"],
.stDateInput input {
    border-radius:10px;
}

.stSelectbox [data-baseweb="select"] *,
.stDateInput input,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    color:#172554 !important;
}

[data-testid="stNumberInput"] input {
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
}

[data-testid="stAppViewContainer"] .stAlert p {
    color:#172554 !important;
}

.hero {
    background:linear-gradient(120deg,#172554 0%,#273f8f 55%,#4263eb 100%);
    border-radius:22px;
    padding:1.55rem 1.8rem;
    margin-bottom:1.25rem;
    color:white;
    box-shadow:0 14px 35px rgba(39,63,143,.18);
}

.hero h1 {
    color:#ffffff !important;
    margin:0;
    font-size:2.15rem;
    letter-spacing:-.05em;
}

.hero p {
    color:#dce6ff !important;
    margin:.35rem 0 0;
    font-size:.98rem;
}

.hero-kicker {
    color:#b8caff !important;
    text-transform:uppercase;
    font-size:.7rem;
    font-weight:800;
    letter-spacing:.16em;
    margin-bottom:.35rem;
}

.section-head {
    display:flex;
    align-items:center;
    gap:.55rem;
    margin:1.25rem 0 .7rem;
    color:var(--ink);
    font-family:Arial,sans-serif;
    font-size:1.12rem;
    font-weight:800;
    letter-spacing:-.025em;
}

.section-head:before {
    content:'';
    display:block;
    width:5px;
    height:22px;
    border-radius:5px;
    background:var(--blue);
}

.note {
    background:#eef4ff;
    color:#274690;
    border:1px solid #d8e5ff;
    border-radius:12px;
    padding:.8rem 1rem;
    font-size:.88rem;
    line-height:1.5;
    margin:.35rem 0 .8rem;
}

.note-warn {
    background:#fff5f3;
    color:#9b2c2c;
    border-color:#ffd8d2;
}

.small-muted {
    color:var(--muted);
    font-size:.83rem;
}


[data-testid="stAppViewContainer"] .js-plotly-plot text {
    fill:#334155 !important;
}

[data-testid="stAppViewContainer"] .js-plotly-plot .xtick text,
[data-testid="stAppViewContainer"] .js-plotly-plot .ytick text,
[data-testid="stAppViewContainer"] .js-plotly-plot .g-xtitle text,
[data-testid="stAppViewContainer"] .js-plotly-plot .g-ytitle text,
[data-testid="stAppViewContainer"] .js-plotly-plot .legend text {
    fill:#334155 !important;
}

[data-testid="stAppViewContainer"] .js-plotly-plot .gridlayer path {
    stroke:#d5dce8 !important;
}

[data-testid="stAppViewContainer"] .js-plotly-plot .zerolinelayer path {
    stroke:#94a3b8 !important;


    button[role="tab"],
button[role="tab"] *,
[data-baseweb="tab"],
[data-baseweb="tab"] *,
[data-testid="stTab"],
[data-testid="stTab"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

button[role="tab"] {
    background: #ffff00 !important;
}
}



#MainMenu, footer {
    visibility:hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
html body .stApp [role="tablist"] button[role="tab"],
html body .stApp [role="tablist"] button[role="tab"] *,
html body .stApp [data-testid="stTabs"] button[role="tab"],
html body .stApp [data-testid="stTabs"] button[role="tab"] *,
html body .stApp [role="tablist"] button[role="tab"]:hover,
html body .stApp [role="tablist"] button[role="tab"]:hover *,
html body .stApp [role="tablist"] button[role="tab"][aria-selected="true"],
html body .stApp [role="tablist"] button[role="tab"][aria-selected="true"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    background-clip: border-box !important;
    -webkit-background-clip: border-box !important;
    text-shadow: none !important;
    opacity: 1 !important;
    visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp button[role="tab"] p,
.stApp button[role="tab"] span,
.stApp button[role="tab"] div,
.stApp button[role="tab"][aria-selected="true"] p,
.stApp button[role="tab"][aria-selected="true"] div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    order_path = ROOT / "data" / "processed" / "cleaned_orders.csv"
    item_path = ROOT / "data" / "processed" / "item_level.csv"

    if not order_path.exists() or not item_path.exists():
        return None, None

    orders = pd.read_csv(
        order_path,
        parse_dates=["order_purchase_timestamp"]
    )
    items = pd.read_csv(
        item_path,
        parse_dates=["order_purchase_timestamp"]
    )

    return orders, items


def style_chart(fig, height=360, title=None):
    fig.update_layout(
        height=height,
        title=(
            dict(
                text=title,
                x=0.02,
                xanchor="left",
                font=dict(size=16, color=NAVY)
            ) if title else None
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            size=12,
            color="#334155"
        ),
        margin=dict(
            l=18,
            r=18,
            t=55 if title else 25,
            b=20
        ),
        colorway=PALETTE,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color="#334155", size=12)
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#cbd5e1",
            zeroline=False,
            tickfont=dict(color="#334155", size=11),
            title_font=dict(color="#334155", size=12)
        ),
        yaxis=dict(
            gridcolor="#dbe2ec",
            linecolor="#cbd5e1",
            zeroline=False,
            tickfont=dict(color="#334155", size=11),
            title_font=dict(color="#334155", size=12)
        ),
    )

    fig.update_annotations(
        font=dict(color="#334155", size=11)
    )

    return fig


def section(title):
    st.markdown(
        f"<div class='section-head'>{title}</div>",
        unsafe_allow_html=True
    )


def note(message, warning=False):
    cls = "note note-warn" if warning else "note"
    st.markdown(
        f"<div class='{cls}'>{message}</div>",
        unsafe_allow_html=True
    )


orders, items = load_data()

with st.sidebar:
    st.markdown("# DeliveryGuard")
    st.caption("E-COMMERCE DELIVERY ANALYSIS")
    st.divider()

    st.markdown("### Start here")
    st.caption("Use these steps to explore the data.")

    st.markdown("""
    1. Set the **purchase date range** below.
    2. Choose a **customer state**, or leave all states selected.
    3. Open a tab to answer a specific question.
    4. Hover over a chart to see exact values. Open data tables for details.
    """)

    st.markdown("### Filter guide")
    st.caption(
        "Filters apply across the dashboard. The date range refers to when "
        "an order was placed, not when it was delivered."
    )
    st.divider()

    if orders is not None and not orders.empty:
        min_date = orders["order_purchase_timestamp"].min().date()
        max_date = orders["order_purchase_timestamp"].max().date()

        date_range = st.date_input(
            "Purchase date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        states = ["All states"]

        if "customer_state" in orders.columns:
            states += sorted(
                orders["customer_state"].dropna().astype(str).unique().tolist()
            )

        selected_state = st.selectbox("Customer state", states)

        st.divider()
        st.markdown("### Data in this view")
        st.caption(f"{len(orders):,} delivered orders")
        st.caption(f"{len(items):,} order items")
        st.caption("Olist Brazilian E-Commerce")

    else:
        date_range = ()
        selected_state = "All states"
        st.info("Processed data has not been found yet.")


st.markdown("""
<div class="hero">
  <div class="hero-kicker">Delivery performance analysis</div>
  <h1>DeliveryGuard</h1>
  <p>Explore delivery times, compare late-delivery rates, and test a simple prediction model using historical orders.</p>
</div>
""", unsafe_allow_html=True)

if orders is None or items is None:
    st.warning(
        "Processed datasets are missing. Run the data pipeline from the project folder:"
    )
    st.code("python -m src.data_preprocessing", language="powershell")
    st.stop()


if len(date_range) == 2:
    start_date = pd.Timestamp(date_range[0])
    end_exclusive = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

    order_filtered = orders[
        orders["order_purchase_timestamp"].ge(start_date)
        & orders["order_purchase_timestamp"].lt(end_exclusive)
    ].copy()

    item_filtered = items[
        items["order_purchase_timestamp"].ge(start_date)
        & items["order_purchase_timestamp"].lt(end_exclusive)
    ].copy()

else:
    order_filtered, item_filtered = orders.copy(), items.copy()


if selected_state != "All states" and "customer_state" in order_filtered.columns:
    order_filtered = order_filtered[
        order_filtered["customer_state"].astype(str) == selected_state
    ]
    item_filtered = item_filtered[
        item_filtered["order_id"].isin(order_filtered["order_id"])
    ]


if order_filtered.empty:
    st.warning(
        "No orders match these filters. Change the date range or state in the sidebar."
    )
    st.stop()


filter_signature = (
    str(date_range[0]) if len(date_range) == 2 else "all",
    str(date_range[1]) if len(date_range) == 2 else "all",
    selected_state,
    len(order_filtered),
)

if st.session_state.get("model_filter_signature") != filter_signature:
    st.session_state.model_result = None
    st.session_state.model_filter_signature = filter_signature


st.caption(
    f"Showing **{len(order_filtered):,} orders** and "
    f"**{len(item_filtered):,} items** for the selected filters."
)

st.markdown("### How to explore the results")
st.markdown(
    "Start with **Overview** to understand the overall delivery pattern. "
    "Use **Category analysis** and **Seller analysis** to compare groups, "
    "**Trends** to compare time periods, and **Try a prediction** to "
    "experiment with the optional model."
)

with st.expander("Metric definitions and how to read the charts", expanded=False):
    st.markdown("""
    - **Late delivery rate:** the percentage of analysed orders delivered after the estimated date.
    - **Delivery time:** days between purchase and delivery.
    - **Early / on time / late:** delivery before, on, or after the estimated date.
    - **Category results:** based on item records, so an order with multiple items can contribute to more than one category.
    - **Seller and state results:** comparisons, not proof that a seller or location caused a delay.

    The date filter uses the **purchase date**, not the delivery date. All tabs use the filters in the sidebar.
    """)


tab_overview, tab_category, tab_sellers, tab_temporal, tab_predict = st.tabs([
    "Overview",
    "Category analysis",
    "Seller analysis",
    "Trends",
    "Try a prediction"
])

import streamlit.components.v1 as components

components.html("""
<script>
const fix = () => {
  const doc = window.parent.document;
  doc.querySelectorAll('[role="tab"], [role="tab"] *').forEach(el => {
    el.style.setProperty('color', '#000000', 'important');
    el.style.setProperty('-webkit-text-fill-color', '#000000', 'important');
  });
};
fix();
setInterval(fix, 500);
</script>
""", height=0)

with tab_overview:
    st.markdown("### What this tab shows")
    st.caption(
        "A summary of delivery performance for the selected orders. "
        "Use the first four figures for a quick overview, then check the "
        "charts to understand the spread of delivery outcomes."
    )

    stats = analytics.compute_overview(order_filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders analysed", f"{stats['total_orders']:,}")
    c2.metric(
        "Late delivery rate",
        f"{stats['late_rate_pct']:.2f}%",
        delta=f"{stats['late_orders']:,} late orders",
        delta_color="inverse"
    )
    c3.metric("Average delivery", f"{stats['avg_delivery_days']:.1f} days")
    c4.metric("Median delivery", f"{stats['median_delivery_days']:.1f} days")

    section("Delivery outcomes")
    left, right = st.columns([1, 1.6])

    with left:
        acc = analytics.accuracy_summary(order_filtered)

        if not acc.empty:
            fig = px.pie(
                acc,
                names="bucket",
                values="count",
                hole=.64,
                color="bucket",
                color_discrete_map={
                    "Early": GREEN,
                    "On time": BLUE,
                    "Late": RED
                },
            )

            fig.update_traces(
                textinfo="percent",
                textposition="inside",
                insidetextfont=dict(color="white", size=12)
            )

            style_chart(fig, 330, "When did orders arrive?")
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", y=-.08)
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                acc.rename(columns={
                    "bucket": "Outcome",
                    "count": "Orders",
                    "pct": "Share (%)",
                    "avg_delay_days": "Avg. days vs estimate"
                }),
                hide_index=True,
                use_container_width=True
            )

    with right:
        fig = px.histogram(
            order_filtered,
            x="delivery_days",
            nbins=42,
            color_discrete_sequence=[BLUE]
        )

        fig.add_vline(
            x=stats["avg_delivery_days"],
            line_dash="dash",
            line_color=RED,
            annotation_text="Mean"
        )
        fig.add_vline(
            x=stats["median_delivery_days"],
            line_dash="dot",
            line_color=GREEN,
            annotation_text="Median"
        )

        fig.update_layout(
            xaxis_title="Days from purchase to delivery",
            yaxis_title="Number of orders"
        )

        style_chart(fig, 390, "How long deliveries take")
        st.plotly_chart(fig, use_container_width=True)

        delay = analytics.delay_distribution(order_filtered)

        fig = px.histogram(
            x=delay,
            nbins=50,
            color_discrete_sequence=["#7C8FCF"]
        )

        fig.add_vline(
            x=0,
            line_color=RED,
            line_width=2,
            annotation_text="Estimated date"
        )

        fig.update_layout(
            xaxis_title="Days early (−) or late (+)",
            yaxis_title="Orders"
        )

        style_chart(fig, 290, "Actual delivery compared with estimate")
        st.plotly_chart(fig, use_container_width=True)

    note(
        f"<b>{stats['pct_early']:.1f}%</b> arrived early, "
        f"<b>{stats['pct_on_time']:.1f}%</b> arrived on the estimated date, "
        f"and <b>{stats['pct_late']:.1f}%</b> arrived late. "
        "These are historical associations, not proof of what caused delays."
    )


with tab_category:
    st.markdown("### What this tab shows")
    st.caption(
        "Compare late-delivery rates and delivery times across product categories. "
        "Adjust the minimum item count to exclude categories with very little data; "
        "use the sample-size chart to judge how much evidence supports each rate."
    )

    section("Which categories have more delays?")
    a, b = st.columns([1, 1])

    min_items = a.slider(
        "Minimum items per category", 10, 1000, 100, 10, key="min_items"
    )
    a.caption("Categories with fewer items are hidden.")

    top_n = b.slider(
        "Categories to display", 5, 30, 18, key="cat_top_n"
    )
    b.caption("Controls how many categories appear in the charts.")

    cat = analytics.late_rate_by_category(
        item_filtered,
        min_items=min_items,
        top_n=top_n
    )

    if cat.empty:
        st.info(
            "No categories meet that item-count threshold. Lower the minimum to see more results."
        )

    else:
        fig = px.bar(
            cat.sort_values("late_rate_pct"),
            x="late_rate_pct",
            y="category",
            orientation="h",
            color="late_rate_pct",
            color_continuous_scale=["#b8e0c4", "#f4d06f", "#e5484d"],
            custom_data=["total_items", "late_items"],
            labels={
                "late_rate_pct": "Late rate (%)",
                "category": ""
            }
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>Late rate: %{x:.1f}%"
                "<br>Items: %{customdata[0]:,}"
                "<br>Late items: %{customdata[1]:,}<extra></extra>"
            )
        )
        fig.update_coloraxes(showscale=False)

        style_chart(
            fig,
            max(360, len(cat) * 25),
            "Late rate by category"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            cat,
            x="total_items",
            y="late_rate_pct",
            size="total_items",
            hover_name="category",
            color="late_rate_pct",
            color_continuous_scale=["#b8e0c4", "#f4d06f", "#e5484d"],
            labels={
                "total_items": "Items observed",
                "late_rate_pct": "Late rate (%)"
            }
        )

        fig.update_coloraxes(showscale=False)
        style_chart(fig, 350, "Late rate alongside sample size")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View category data"):
            st.dataframe(
                cat.rename(columns={
                    "category": "Category",
                    "total_items": "Items",
                    "late_items": "Late items",
                    "late_rate_pct": "Late rate (%)"
                }),
                hide_index=True,
                use_container_width=True
            )

    section("Delivery speed by category")

    speed = analytics.delivery_days_by_category(
        item_filtered,
        min_items=min_items
    )

    if speed.empty:
        st.info("No category delivery-speed results for these settings.")

    else:
        speed = speed.head(top_n).sort_values("avg_days")

        fig = px.bar(
            speed,
            x="avg_days",
            y="category",
            orientation="h",
            color="avg_days",
            color_continuous_scale=["#b8e0c4", "#f4d06f", "#e5484d"],
            labels={
                "avg_days": "Average delivery (days)",
                "category": ""
            }
        )

        fig.update_coloraxes(showscale=False)
        style_chart(
            fig,
            max(320, len(speed) * 24),
            "Average delivery time by category"
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_sellers:
    st.markdown("### What this tab shows")
    st.caption(
        "Compare seller late rates and customer-state results. Set a minimum "
        "number of orders so sellers with only a few observations do not "
        "dominate the ranking. These comparisons do not prove that a seller "
        "or state caused a delay."
    )

    section("Seller delivery performance")
    a, b = st.columns(2)

    min_orders = a.slider(
        "Minimum orders per seller", 5, 200, 20, 5, key="min_orders"
    )
    a.caption("Sellers below this order count are excluded.")

    seller_top_n = b.slider(
        "Sellers to display", 5, 30, 15, key="seller_top_n"
    )
    b.caption("Controls the number of sellers shown.")

    seller_stats = analytics.late_rate_by_seller(
        order_filtered,
        min_orders=min_orders,
        top_n=seller_top_n
    )

    if seller_stats.empty:
        st.info("No sellers meet this minimum order count. Try lowering the threshold.")

    else:
        seller_stats = seller_stats.copy()
        seller_stats["seller_label"] = (
            seller_stats["primary_seller_id"].astype(str).str[:8] + "…"
        )

        fig = px.bar(
            seller_stats.sort_values("late_rate_pct"),
            x="late_rate_pct",
            y="seller_label",
            orientation="h",
            color="late_rate_pct",
            color_continuous_scale=["#b8e0c4", "#f4d06f", "#e5484d"],
            custom_data=["total_orders", "late_orders"],
            labels={
                "late_rate_pct": "Late rate (%)",
                "seller_label": "Seller ID"
            }
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>Late rate: %{x:.1f}%"
                "<br>Orders: %{customdata[0]:,}"
                "<br>Late orders: %{customdata[1]:,}<extra></extra>"
            )
        )
        fig.update_coloraxes(showscale=False)

        style_chart(
            fig,
            max(330, len(seller_stats) * 26),
            "Sellers with the highest late rates"
        )
        st.plotly_chart(fig, use_container_width=True)

        note(
            f"Only sellers with at least {min_orders} orders are included. "
            "Rates from small samples can be misleading."
        )

    section("Customer-state comparison")

    state_stats = analytics.late_rate_by_state(
        order_filtered,
        col="customer_state"
    )

    if state_stats.empty:
        st.info("Customer-state information is not available for this selection.")

    else:
        fig = px.bar(
            state_stats.sort_values("late_rate_pct"),
            x="customer_state",
            y="late_rate_pct",
            color="late_rate_pct",
            color_continuous_scale=["#b8e0c4", "#f4d06f", "#e5484d"],
            custom_data=["total_orders", "late_orders"],
            labels={
                "customer_state": "State",
                "late_rate_pct": "Late rate (%)"
            }
        )

        fig.update_traces(
            hovertemplate=(
                "State: %{x}<br>Late rate: %{y:.1f}%"
                "<br>Orders: %{customdata[0]:,}"
                "<br>Late orders: %{customdata[1]:,}<extra></extra>"
            )
        )
        fig.update_coloraxes(showscale=False)

        style_chart(fig, 360, "Late-delivery rate by customer state")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View state data"):
            st.dataframe(
                state_stats.rename(columns={
                    "customer_state": "State",
                    "total_orders": "Orders",
                    "late_orders": "Late orders",
                    "late_rate_pct": "Late rate (%)"
                }),
                hide_index=True,
                use_container_width=True
            )


with tab_temporal:
    st.markdown("### What this tab shows")
    st.caption(
        "Compare late-delivery rates over months and weekdays. Monthly order "
        "volume provides context for each rate; a change in the chart does "
        "not explain its cause by itself."
    )

    section("Monthly delivery trends")
    monthly = analytics.late_rate_by_month(order_filtered)

    if monthly.empty:
        st.info("No monthly data is available for these filters.")

    else:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=monthly["year_month"],
                y=monthly["total_orders"],
                name="Orders",
                marker_color="#dce5fb",
                yaxis="y2"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly["year_month"],
                y=monthly["late_rate_pct"],
                name="Late rate",
                mode="lines+markers",
                line=dict(color=BLUE, width=3),
                marker=dict(size=7)
            )
        )

        fig.update_layout(
            yaxis=dict(
                title="Late rate (%)",
                ticksuffix="%"
            ),
            yaxis2=dict(
                title="Orders",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(
                orientation="h",
                y=1.12
            ),
            xaxis_title="Purchase month"
        )

        style_chart(fig, 420, "Late rate and order volume")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.area(
            monthly,
            x="year_month",
            y="total_orders",
            color_discrete_sequence=[BLUE],
            labels={
                "year_month": "Month",
                "total_orders": "Orders"
            }
        )

        fig.update_traces(
            fill="tozeroy",
            fillcolor="rgba(66,99,235,.12)",
            line=dict(width=2)
        )

        style_chart(fig, 280, "Monthly order volume")
        st.plotly_chart(fig, use_container_width=True)

    section("Does the day of purchase matter?")
    dow = analytics.late_rate_by_dayofweek(order_filtered)

    if dow.empty:
        st.info("No weekday data is available.")

    else:
        fig = px.bar(
            dow,
            x="day_name",
            y="late_rate_pct",
            text="late_rate_pct",
            color="late_rate_pct",
            color_continuous_scale=["#b8e0c4", "#f4d06f", "#e5484d"],
            labels={
                "day_name": "Day purchased",
                "late_rate_pct": "Late rate (%)"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )
        fig.update_coloraxes(showscale=False)

        style_chart(fig, 330, "Late-delivery rate by weekday")
        st.plotly_chart(fig, use_container_width=True)


with tab_predict:
    st.markdown("### How to use this tab")
    st.markdown("""
    1. Select **Train model on filtered data**.
    2. Review the evaluation metrics and confusion matrix.
    3. Enter details for a hypothetical order to see the model's estimate.

    The model is for learning and exploration; its prediction is not a delivery guarantee.
    """)

    section("Train a baseline model")

    st.write(
        "The model learns patterns from older orders and is evaluated on later "
        "orders it did not train on. Its estimates can be wrong, especially "
        "for late orders that are uncommon in the dataset."
    )

    if "model_result" not in st.session_state:
        st.session_state.model_result = None

    if st.button("Train model on filtered data", type="primary"):
        with st.spinner("Training and evaluating the model…"):
            st.session_state.model_result = prediction.train_model(order_filtered)

    model_result = st.session_state.model_result

    if model_result is None:
        st.info(
            "Select **Train model on filtered data**. When training finishes, "
            "review the evaluation metrics before trying a hypothetical order."
        )

    elif "error" in model_result:
        st.error(model_result["error"])

    else:
        metrics = model_result["metrics"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precision", f"{metrics['precision']:.3f}")
        c2.metric("Recall", f"{metrics['recall']:.3f}")
        c3.metric("F1 score", f"{metrics['f1']:.3f}")
        c4.metric(
            "ROC–AUC",
            f"{metrics['roc_auc']:.3f}"
            if pd.notna(metrics["roc_auc"]) else "N/A"
        )
        c5.metric("Baseline accuracy", f"{metrics['baseline_accuracy']:.3f}")

        left, right = st.columns(2)

        with left:
            cm = model_result["confusion_matrix"]

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted on time", "Predicted late"],
                y=["Actually on time", "Actually late"],
                color_continuous_scale=["#eef2ff", BLUE],
                aspect="auto"
            )

            fig.update_coloraxes(showscale=False)
            style_chart(fig, 310, "Confusion matrix")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fi = model_result["feature_importance"].copy().sort_values("coefficient")

            fig = px.bar(
                fi,
                x="coefficient",
                y="feature",
                orientation="h",
                color="coefficient",
                color_continuous_scale=[RED, "#dbe4ff", BLUE],
                labels={
                    "coefficient": "Model coefficient",
                    "feature": "Feature"
                }
            )

            fig.update_coloraxes(showscale=False, cmid=0)
            style_chart(fig, 310, "Feature relationships")
            st.plotly_chart(fig, use_container_width=True)

        note(
            f"Trained on {model_result['train_size']:,} earlier orders and "
            f"evaluated on {model_result['test_size']:,} later orders. "
            "Precision indicates how many orders flagged late were actually "
            "late; recall indicates how many late orders the model found."
        )

        with st.expander("Classification report"):
            st.code(model_result["classification_report"])

        section("Try a hypothetical order")

        p1, p2, p3 = st.columns(3)

        month = p1.selectbox(
            "Purchase month",
            list(range(1, 13)),
            index=7
        )

        weekday_names = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]

        weekday_name = p1.selectbox(
            "Purchase weekday",
            weekday_names,
            index=1
        )
        weekday = weekday_names.index(weekday_name)

        window = p2.number_input(
            "Estimated delivery window (days)",
            min_value=1.0,
            max_value=90.0,
            value=15.0
        )

        price = p2.number_input(
            "Order value (BRL)",
            min_value=0.0,
            value=150.0,
            step=10.0
        )

        freight = p3.number_input(
            "Freight value (BRL)",
            min_value=0.0,
            value=20.0,
            step=5.0
        )

        item_count = p3.number_input(
            "Number of items",
            min_value=1,
            value=1,
            step=1
        )

        seller_count = p3.number_input(
            "Number of sellers",
            min_value=1,
            value=1,
            step=1
        )

        default_rate = float(order_filtered["is_late"].mean())

        hist_rate = st.slider(
            "Seller's historical late rate (%)",
            0.0,
            100.0,
            float(round(default_rate * 100, 1)),
            0.5,
            help=(
                "Enter a past late-delivery percentage from 0 to 100. "
                "If you do not know the seller's rate, leave the default "
                "as a rough dataset-level fallback."
            )
        ) / 100

        result = prediction.predict_single(
            model_result,
            month,
            weekday,
            window,
            price,
            freight,
            item_count,
            seller_count,
            hist_rate
        )

        if "error" not in result:
            prob = result["probability_late"] * 100

            st.metric(
                "Estimated probability of late delivery",
                f"{prob:.1f}%",
                delta=result["prediction"],
                delta_color="inverse"
            )

            st.progress(
                min(max(result["probability_late"], 0.0), 1.0),
                text="Estimated late-delivery probability"
            )

            st.caption(
                "This percentage is an estimate from historical patterns, "
                "not a guarantee. It should not be used to make real delivery commitments."
            )


st.divider()

st.markdown(
    "<p class='small-muted'>DeliveryGuard | Historical analysis of the "
    "Olist Brazilian e-commerce dataset | Results describe associations, not causes.</p>",
    unsafe_allow_html=True
)