from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# OLIST E-COMMERCE ANALYTICS DASHBOARD
# Data source: dashboard/main_data.csv
# ============================================================

st.set_page_config(
    page_title="Olist | E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme -----------------------------
NAVY = "#14213D"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
MUTED = "#64748B"
GRID = "#E2E8F0"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{ background: #F5F7FB; }}
    [data-testid="stHeader"] {{ background: rgba(245,247,251,0.88); }}
    [data-testid="stSidebar"] {{ background: #101C33; }}
    [data-testid="stSidebar"] * {{ color: #E8EEF8 !important; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: #B8C5D9 !important; }}
    .block-container {{ padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1500px; }}
    h1, h2, h3 {{ font-family: 'Manrope', sans-serif; color: {NAVY}; letter-spacing: -0.03em; }}
    .hero {{
        background: linear-gradient(115deg, #14213D 0%, #1D3B68 58%, #2563EB 100%);
        padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 22px;
        box-shadow: 0 12px 28px rgba(20,33,61,.13);
    }}
    .hero h1 {{ color: white !important; font-size: 30px; margin: 0 0 6px 0; }}
    .hero p {{ color: #D9E5F6; margin: 0; font-size: 14px; }}
    .eyebrow {{ color: #93C5FD; text-transform: uppercase; letter-spacing: .13em;
        font-size: 10px; font-weight: 700; margin-bottom: 9px; }}
    .section-note {{ color: {MUTED}; font-size: 13px; margin-top: -8px; margin-bottom: 14px; }}
    .metric-card {{
        background: white; border: 1px solid #E8EDF5; border-radius: 14px;
        padding: 17px 19px; min-height: 112px; box-shadow: 0 3px 12px rgba(15,23,42,.035);
    }}
    .metric-label {{ color: {MUTED}; font-size: 12px; font-weight: 600; margin-bottom: 8px; }}
    .metric-value {{ color: {NAVY}; font-family: 'Manrope', sans-serif;
        font-size: 25px; font-weight: 800; letter-spacing: -.04em; }}
    .metric-sub {{ color: #94A3B8; font-size: 11px; margin-top: 4px; }}
    .insight {{
        background: white; border: 1px solid #E8EDF5; border-left: 4px solid {BLUE};
        border-radius: 10px; padding: 14px 16px; margin: 8px 0 12px 0;
        color: #334155; font-size: 13px;
    }}
    .small-muted {{ color: {MUTED}; font-size: 12px; }}
    div[data-testid="stTabs"] button {{ font-weight: 600; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid #E8EDF5; border-radius: 10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Helpers -----------------------------
@st.cache_data(show_spinner="Memuat dan menyiapkan data...")
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=["order_purchase_timestamp"])
    required = {
        "order_id", "customer_id", "customer_unique_id",
        "order_purchase_timestamp", "payment_value", "customer_state",
        "latitude", "longitude",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce").fillna(0)
    df["customer_state"] = df["customer_state"].fillna("Unknown").astype(str)
    df = df.dropna(subset=["order_purchase_timestamp", "customer_unique_id"])
    return df


def brl(value, decimals=0):
    return f"R$ {value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def metric_card(label, value, sub=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def polish(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans", color="#334155", size=12),
        margin=dict(l=18, r=18, t=48, b=18),
        title=dict(font=dict(family="Manrope", size=15, color=NAVY), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F7", linecolor="rgba(0,0,0,0)", zeroline=False)
    return fig


def calculate_rfm(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    snapshot = data["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)
    rfm = (
        data.groupby("customer_unique_id")
        .agg(
            Recency=("order_purchase_timestamp", lambda x: (snapshot - x.max().normalize()).days),
            Frequency=("order_id", "nunique"),
            Monetary=("payment_value", "sum"),
        )
        .reset_index()
    )

    # Score definitions follow the analysis notebook.
    rfm["F_Score"] = np.select(
        [rfm["Frequency"].eq(1), rfm["Frequency"].eq(2), rfm["Frequency"].ge(3)],
        [1, 3, 5],
        default=1,
    )
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1],
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    def segment(row):
        if row["R_Score"] >= 4 and row["F_Score"] >= 3 and row["M_Score"] >= 4:
            return "Champions"
        if row["F_Score"] >= 3 and row["R_Score"] >= 3:
            return "Loyal Customers"
        if row["R_Score"] >= 4 and row["F_Score"] == 1:
            return "New Customers"
        if row["R_Score"] >= 3 and row["M_Score"] >= 3:
            return "Potential Loyalists"
        if row["R_Score"] <= 2 and row["F_Score"] >= 3:
            return "At Risk"
        return "Hibernating"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm


def empty_message():
    st.info("Tidak ada data untuk kombinasi filter yang dipilih. Coba perluas rentang tanggal atau pilih lebih banyak negara bagian.")


# ----------------------------- Load -----------------------------
candidate_paths = [
    Path(__file__).resolve().parent / "main_data.csv",
    Path.cwd() / "dashboard" / "main_data.csv",
    Path.cwd() / "main_data.csv",
]
DATA_FILE = next((p for p in candidate_paths if p.exists()), None)

if DATA_FILE is None:
    st.error(
        "File `main_data.csv` tidak ditemukan. Pastikan file berada satu folder "
        "dengan `dashboard.py` di dalam folder `dashboard/`."
    )
    st.stop()

try:
    raw = load_data(str(DATA_FILE))
except Exception as exc:
    st.error(f"Gagal membaca data: {exc}")
    st.stop()

if raw.empty:
    st.error("Dataset tidak memiliki baris valid.")
    st.stop()

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## OLIST ANALYTICS")
    st.markdown("<p class='small-muted'>Brazilian E-Commerce Public Dataset</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "NAVIGASI",
        ["Overview", "Sales Performance", "Customer RFM", "Geographic Analysis"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("### FILTER DATA")
    min_date = raw["order_purchase_timestamp"].min().date()
    max_date = raw["order_purchase_timestamp"].max().date()
    date_range = st.date_input(
        "Periode transaksi",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    states = sorted(raw["customer_state"].dropna().unique().tolist())
    selected_states = st.multiselect("Negara bagian", states, default=states)
    st.markdown("---")
    st.caption("Data mencakup pesanan berstatus delivered. Nilai pembayaran dalam Brazilian Real (BRL).")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered = raw[
    raw["order_purchase_timestamp"].dt.date.between(start_date, end_date)
    & raw["customer_state"].isin(selected_states)
].copy()

if filtered.empty:
    st.markdown("<div class='hero'><div class='eyebrow'>Brazilian E-Commerce Intelligence</div><h1>Olist Analytics Dashboard</h1><p>Analisis performa penjualan, perilaku pelanggan, dan distribusi geografis.</p></div>", unsafe_allow_html=True)
    empty_message()
    st.stop()

# ----------------------------- Header -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Brazilian E-Commerce Intelligence</div>
      <h1>Olist Analytics Dashboard</h1>
      <p>Sales performance · Customer behavior · Geographic distribution</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='small-muted'>Periode aktif: <b>{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}</b>"
    f" &nbsp;|&nbsp; {len(selected_states)} negara bagian dipilih</div>",
    unsafe_allow_html=True,
)
st.write("")

# Shared aggregations
orders_count = filtered["order_id"].nunique()
customers_count = filtered["customer_unique_id"].nunique()
total_payment = filtered.groupby("order_id")["payment_value"].first().sum()
avg_payment = filtered.groupby("order_id")["payment_value"].first().mean()
monthly = (
    filtered.assign(Month=filtered["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp())
    .groupby("Month")
    .agg(Orders=("order_id", "nunique"), Payment=("payment_value", "sum"))
    .reset_index()
)
rfm = calculate_rfm(filtered)

# ----------------------------- OVERVIEW -----------------------------
if page == "Overview":
    st.subheader("Executive Overview")
    st.markdown("<div class='section-note'>Ringkasan indikator utama berdasarkan filter aktif.</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]: metric_card("DELIVERED ORDERS", f"{orders_count:,.0f}", "Jumlah pesanan unik")
    with cols[1]: metric_card("UNIQUE CUSTOMERS", f"{customers_count:,.0f}", "Pelanggan unik")
    with cols[2]: metric_card("TOTAL PAYMENT", brl(total_payment), "Akumulasi nilai pembayaran")
    with cols[3]: metric_card("AVERAGE PAYMENT / ORDER", brl(avg_payment, 2), "Rata-rata per pesanan")
    st.write("")
    left, right = st.columns([1.55, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["Month"], y=monthly["Orders"], name="Delivered orders",
            marker_color=BLUE, yaxis="y",
            hovertemplate="%{x|%b %Y}<br>Orders: %{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Payment"], name="Payment (BRL)",
            mode="lines+markers", line=dict(color=ORANGE, width=3),
            marker=dict(size=6), yaxis="y2",
            hovertemplate="%{x|%b %Y}<br>Payment: R$ %{y:,.2f}<extra></extra>",
        ))
        fig.update_layout(
            title="Monthly Sales Performance", height=390, paper_bgcolor="white",
            plot_bgcolor="white", margin=dict(l=20,r=20,t=55,b=20),
            font=dict(family="DM Sans", color="#334155"),
            yaxis=dict(title="Orders", showgrid=True, gridcolor="#EEF2F7"),
            yaxis2=dict(title="Payment (BRL)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if not rfm.empty:
            seg = rfm["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
            fig = px.pie(
                seg, names="Segment", values="Customers", hole=.62,
                title="Customer Segment Mix",
                color_discrete_sequence=[BLUE, TEAL, ORANGE, PURPLE, "#94A3B8", "#D97706"],
            )
            fig.update_traces(textposition="inside", textinfo="percent", hole=.62)
            polish(fig, 390)
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_message()

    st.subheader("Key Findings")
    if not monthly.empty:
        peak = monthly.loc[monthly["Orders"].idxmax()]
        st.markdown(
            f"<div class='insight'><b>Sales peak:</b> bulan {peak['Month'].strftime('%B %Y')} mencatat "
            f"{peak['Orders']:,.0f} delivered orders pada periode filter aktif.</div>",
            unsafe_allow_html=True,
        )
    if not rfm.empty:
        top_seg = rfm["Segment"].value_counts().idxmax()
        top_share = rfm["Segment"].value_counts(normalize=True).max() * 100
        st.markdown(
            f"<div class='insight'><b>Customer profile:</b> segmen terbesar adalah "
            f"<b>{top_seg}</b>, mencakup {top_share:.1f}% pelanggan pada filter aktif.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='insight'><b>Interpretasi:</b> gunakan tren bulanan untuk memantau dinamika penjualan, "
        "segmentasi RFM untuk menentukan prioritas retensi, dan distribusi negara bagian untuk memahami "
        "konsentrasi pasar. Angka volume tidak secara langsung menunjukkan profitabilitas atau potensi pertumbuhan.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------- SALES -----------------------------
elif page == "Sales Performance":
    st.subheader("Sales Performance")
    st.markdown("<div class='section-note'>Tren pesanan delivered dan nilai pembayaran bulanan.</div>", unsafe_allow_html=True)
    if monthly.empty:
        empty_message()
    else:
        peak = monthly.loc[monthly["Orders"].idxmax()]
        low = monthly.loc[monthly["Orders"].idxmin()]
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("PEAK MONTH", peak["Month"].strftime("%b %Y"), f"{peak['Orders']:,.0f} orders")
        with c2: metric_card("LOWEST ACTIVE MONTH", low["Month"].strftime("%b %Y"), f"{low['Orders']:,.0f} orders")
        with c3: metric_card("MONTHLY AVERAGE", f"{monthly['Orders'].mean():,.0f}", "Delivered orders")
        fig1 = px.bar(
            monthly, x="Month", y="Orders", title="Monthly Delivered Orders",
            labels={"Month": "Month", "Orders": "Delivered orders"},
            color_discrete_sequence=[BLUE],
        )
        fig1.update_traces(hovertemplate="%{x|%B %Y}<br>%{y:,.0f} orders<extra></extra>")
        polish(fig1, 370)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.area(
            monthly, x="Month", y="Payment", title="Monthly Payment Value",
            labels={"Month": "Month", "Payment": "Payment (BRL)"},
            color_discrete_sequence=[TEAL],
        )
        fig2.update_traces(line=dict(width=2), fillcolor="rgba(15,118,110,.14)")
        fig2.update_yaxes(tickprefix="R$ ")
        polish(fig2, 370)
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Lihat tabel performa bulanan"):
            table = monthly.copy()
            table["Month"] = table["Month"].dt.strftime("%B %Y")
            table["Payment (BRL)"] = table["Payment"].map(lambda x: brl(x, 2))
            table = table.rename(columns={"Month": "Month", "Orders": "Delivered Orders"})
            st.dataframe(table[["Month", "Delivered Orders", "Payment (BRL)"]], use_container_width=True, hide_index=True)

# ----------------------------- CUSTOMER RFM -----------------------------
elif page == "Customer RFM":
    st.subheader("Customer Segmentation — RFM Analysis")
    st.markdown(
        "<div class='section-note'>Recency, Frequency, dan Monetary dihitung dari pesanan delivered pada periode dan negara bagian yang dipilih.</div>",
        unsafe_allow_html=True,
    )
    if rfm.empty:
        empty_message()
    else:
        segment_order = ["Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "At Risk", "Hibernating"]
        rfm_summary = (
            rfm.groupby("Segment")
            .agg(
                Customers=("customer_unique_id", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean"),
                Total_Monetary=("Monetary", "sum"),
            )
            .reindex(segment_order)
            .dropna()
            .reset_index()
        )
        rfm_summary["Customer Share (%)"] = rfm_summary["Customers"] / rfm["customer_unique_id"].nunique() * 100
        rfm_summary["Monetary Contribution (%)"] = rfm_summary["Total_Monetary"] / rfm["Monetary"].sum() * 100

        a, b, c = st.columns(3)
        with a: metric_card("CUSTOMERS ANALYZED", f"{len(rfm):,.0f}", "Unique customers")
        with b: metric_card("AVG. RECENCY", f"{rfm['Recency'].mean():.0f} days", "Days since last purchase")
        with c: metric_card("REPEAT CUSTOMERS", f"{(rfm['Frequency'] > 1).mean()*100:.2f}%", "Frequency greater than one")
        st.write("")
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                rfm_summary.sort_values("Customers", ascending=True),
                x="Customers", y="Segment", orientation="h",
                title="Customers by RFM Segment",
                color="Customers", color_continuous_scale=["#BFDBFE", BLUE],
                text="Customers",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_coloraxes(showscale=False)
            polish(fig, 410)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                rfm_summary.sort_values("Monetary Contribution (%)", ascending=True),
                x="Monetary Contribution (%)", y="Segment", orientation="h",
                title="Monetary Contribution by Segment",
                color="Monetary Contribution (%)", color_continuous_scale=["#CCFBF1", TEAL],
                text="Monetary Contribution (%)",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_coloraxes(showscale=False)
            polish(fig, 410)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Segment Profile")
        display_summary = rfm_summary.copy()
        display_summary["Avg. Monetary"] = display_summary["Avg_Monetary"].map(lambda x: brl(x, 2))
        display_summary["Total Monetary"] = display_summary["Total_Monetary"].map(lambda x: brl(x, 2))
        display_summary["Avg. Recency (days)"] = display_summary["Avg_Recency"].round(1)
        display_summary["Avg. Frequency"] = display_summary["Avg_Frequency"].round(2)
        display_summary["Customer Share"] = display_summary["Customer Share (%)"].map(lambda x: f"{x:.2f}%")
        display_summary["Monetary Contribution"] = display_summary["Monetary Contribution (%)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(
            display_summary[[
                "Segment", "Customers", "Customer Share", "Avg. Recency (days)",
                "Avg. Frequency", "Avg. Monetary", "Total Monetary", "Monetary Contribution",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Retention Priorities")
        recommendations = {
            "Hibernating": "Prioritaskan kampanye reaktivasi yang terukur; uji penawaran personal dan komunikasi win-back.",
            "New Customers": "Bangun pengalaman pascapembelian dan dorong pembelian kedua melalui rekomendasi produk atau insentif yang relevan.",
            "At Risk": "Lakukan win-back lebih awal dengan pesan yang dipersonalisasi berdasarkan riwayat pembelian.",
            "Champions": "Pertahankan loyalitas melalui program apresiasi, akses awal, atau benefit eksklusif.",
            "Loyal Customers": "Dorong retensi dan cross-selling yang relevan tanpa mengurangi pengalaman pelanggan.",
            "Potential Loyalists": "Arahkan pelanggan yang mulai aktif menuju pembelian berulang melalui rekomendasi dan follow-up.",
        }
        for _, row in rfm_summary.sort_values("Customers", ascending=False).iterrows():
            st.markdown(
                f"<div class='insight'><b>{row['Segment']}</b> · {row['Customers']:,.0f} customers "
                f"({row['Customer Share (%)']:.1f}%)<br>{recommendations[row['Segment']]}</div>",
                unsafe_allow_html=True,
            )
        st.caption("Catatan: skor RFM dan segmentasi dihitung ulang mengikuti filter aktif. Hasilnya dapat berbeda dari baseline notebook jika periode atau negara bagian difilter.")

# ----------------------------- GEOGRAPHIC -----------------------------
elif page == "Geographic Analysis":
    st.subheader("Geographic Distribution")
    st.markdown(
        "<div class='section-note'>Distribusi pelanggan dan pesanan delivered berdasarkan negara bagian pelanggan (customer_state).</div>",
        unsafe_allow_html=True,
    )
    geo = (
        filtered.groupby("customer_state")
        .agg(
            Customers=("customer_unique_id", "nunique"),
            Orders=("order_id", "nunique"),
            Payment=("payment_value", "sum"),
            Latitude=("latitude", "mean"),
            Longitude=("longitude", "mean"),
        )
        .reset_index()
        .sort_values("Orders", ascending=False)
    )
    if geo.empty:
        empty_message()
    else:
        top = geo.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("LEADING STATE", top["customer_state"], f"{top['Orders']:,.0f} delivered orders")
        with c2: metric_card("CUSTOMERS IN LEADING STATE", f"{top['Customers']:,.0f}", "Unique customers")
        with c3: metric_card("STATES IN FILTER", f"{len(geo)}", "Brazilian states / DF")
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                geo.sort_values("Customers", ascending=True),
                x="Customers", y="customer_state", orientation="h",
                title="Unique Customers by State",
                labels={"customer_state": "State", "Customers": "Unique customers"},
                color_discrete_sequence=[BLUE],
            )
            polish(fig, 560)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                geo.sort_values("Orders", ascending=True),
                x="Orders", y="customer_state", orientation="h",
                title="Delivered Orders by State",
                labels={"customer_state": "State", "Orders": "Delivered orders"},
                color_discrete_sequence=[TEAL],
            )
            polish(fig, 560)
            st.plotly_chart(fig, use_container_width=True)

        map_data = geo.dropna(subset=["Latitude", "Longitude"])
        if not map_data.empty:
            fig = px.scatter_geo(
                map_data,
                lat="Latitude",
                lon="Longitude",
                size="Orders",
                color="Orders",
                hover_name="customer_state",
                hover_data={
                    "Customers": ":,",
                    "Orders": ":,",
                    "Payment": ":,.2f",
                    "Latitude": False,
                    "Longitude": False,
                },
                title="Geographic Concentration by Customer State",
                color_continuous_scale="Blues",
                projection="natural earth",
                size_max=34,
            )
            fig.update_geos(
                scope="south america",
                showcountries=True,
                countrycolor="#CBD5E1",
                showland=True,
                landcolor="#F1F5F9",
                showocean=True,
                oceancolor="#F8FAFC",
                showcoastlines=True,
                coastlinecolor="#CBD5E1",
                center=dict(lat=-14, lon=-52),
                lataxis_range=[-35, 6],
                lonaxis_range=[-75, -32],
            )
            polish(fig, 520)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### State-Level Summary")
        geo_display = geo.copy()
        geo_display["Payment (BRL)"] = geo_display["Payment"].map(lambda x: brl(x, 2))
        geo_display = geo_display.rename(columns={
            "customer_state": "State",
            "Customers": "Unique Customers",
            "Orders": "Delivered Orders",
        })
        st.dataframe(
            geo_display[["State", "Unique Customers", "Delivered Orders", "Payment (BRL)"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Peta menggunakan rata-rata koordinat geolocation per negara bagian sebagai titik representatif, bukan batas administratif atau lokasi setiap pelanggan.")

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.markdown(
    "<div class='small-muted'>Olist E-Commerce Analytics · Data source: Brazilian E-Commerce Public Dataset by Olist · "
    "Analisis mencakup pesanan berstatus delivered.</div>",
    unsafe_allow_html=True,
)
