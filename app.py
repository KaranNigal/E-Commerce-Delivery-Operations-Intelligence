import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce Delivery Operations Intelligence",
    page_icon="🚚",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ORDERS_FILE = (
    BASE_DIR
    / "data"
    / "dashboard"
    / "dashboard_orders.csv"
)

FORECAST_FILE = (
    BASE_DIR
    / "models"
    / "demand_forecast_results.csv"
)

CAPACITY_FILE = (
    BASE_DIR
    / "reports"
    / "tables"
    / "capacity_plan.csv"
)

INTELLIGENCE_FILE = (
    BASE_DIR
    / "reports"
    / "tables"
    / "operational_intelligence_dashboard_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_orders():
    df = pd.read_csv(
        ORDERS_FILE,
        low_memory=False,
        keep_default_na=False,
        parse_dates=["Order_Date"],
    )

    return df


@st.cache_data
def load_forecast():
    df = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["Order_Date"],
    )

    return df


@st.cache_data
def load_capacity():
    df = pd.read_csv(
        CAPACITY_FILE,
        parse_dates=["Date"],
    )

    return df


@st.cache_data
def load_intelligence():
    df = pd.read_csv(
        INTELLIGENCE_FILE,
        parse_dates=["Date"],
    )

    return df


# ============================================================
# LOAD DATASETS
# ============================================================

orders = load_orders()
forecast = load_forecast()
capacity = load_capacity()
intelligence = load_intelligence()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚚 Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "🏠 Executive Overview",
        "📈 Demand Forecasting",
        "🚚 Delivery Performance",
        "👥 Capacity Planning",
        "⚠️ Operational Intelligence",
    ],
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.title("🚚 E-Commerce Delivery Operations Intelligence")

    st.markdown(
        """
        End-to-end analytics platform for monitoring delivery operations,
        forecasting demand, analyzing performance, and planning capacity.
        """
    )

    st.markdown("---")

    # KPIs

    total_orders = len(orders)

    avg_delivery_time = orders["Delivery_Time"].mean()

    avg_distance = orders["Distance_KM"].mean()

    unique_cities = orders["City"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col2.metric(
        "Avg Delivery Time",
        f"{avg_delivery_time:.1f} min"
    )

    col3.metric(
        "Avg Delivery Distance",
        f"{avg_distance:.1f} km"
    )

    col4.metric(
        "Cities Covered",
        unique_cities
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Daily demand

    daily_orders = (
        orders
        .groupby("Order_Date")
        .size()
        .reset_index(name="Orders")
    )

    fig_demand = px.line(
        daily_orders,
        x="Order_Date",
        y="Orders",
        title="Daily Order Demand",
    )

    col1.plotly_chart(
        fig_demand,
        use_container_width=True
    )

    # Delivery performance by traffic

    traffic_performance = (
        orders
        .groupby("Traffic")["Delivery_Time"]
        .mean()
        .reset_index()
    )

    fig_traffic = px.bar(
        traffic_performance,
        x="Traffic",
        y="Delivery_Time",
        title="Average Delivery Time by Traffic",
    )

    col2.plotly_chart(
        fig_traffic,
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # City performance

    city_performance = (
        orders
        .groupby("City")["Delivery_Time"]
        .mean()
        .sort_values()
        .reset_index()
    )

    fig_city = px.bar(
        city_performance,
        x="City",
        y="Delivery_Time",
        title="Delivery Performance by City",
    )

    col1.plotly_chart(
        fig_city,
        use_container_width=True
    )

    # Weather impact

    weather_performance = (
        orders
        .groupby("Weather")["Delivery_Time"]
        .mean()
        .sort_values()
        .reset_index()
    )

    fig_weather = px.bar(
        weather_performance,
        x="Weather",
        y="Delivery_Time",
        title="Weather Impact on Delivery Time",
    )

    col2.plotly_chart(
        fig_weather,
        use_container_width=True
    )


# ============================================================
# DEMAND FORECASTING
# ============================================================

elif page == "📈 Demand Forecasting":

    st.title("📈 Demand Forecasting")

    st.markdown(
        "Comparison between actual demand and machine learning predictions."
    )

    st.markdown("---")

    fig_forecast = px.line(
        forecast,
        x="Order_Date",
        y=[
            "Actual_Demand",
            "Predicted_Demand"
        ],
        title="Actual vs Predicted Demand",
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    forecast["Absolute_Error"] = (
        forecast["Actual_Demand"]
        - forecast["Predicted_Demand"]
    ).abs()

    mae = forecast["Absolute_Error"].mean()

    mape = (
        forecast["Absolute_Error"]
        / forecast["Actual_Demand"]
    ).mean() * 100

    col1, col2 = st.columns(2)

    col1.metric(
        "Mean Absolute Error",
        f"{mae:.1f} orders"
    )

    col2.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

    st.markdown("---")

    st.subheader("Forecast Dataset")

    st.dataframe(
        forecast,
        use_container_width=True
    )


# ============================================================
# DELIVERY PERFORMANCE
# ============================================================

elif page == "🚚 Delivery Performance":

    st.title("🚚 Delivery Performance Analysis")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Delivery Time",
        f"{orders['Delivery_Time'].mean():.1f} min"
    )

    col2.metric(
        "Median Delivery Time",
        f"{orders['Delivery_Time'].median():.1f} min"
    )

    col3.metric(
        "Longest Delivery",
        f"{orders['Delivery_Time'].max():.0f} min"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Traffic

    traffic = (
        orders
        .groupby("Traffic")["Delivery_Time"]
        .mean()
        .sort_values()
        .reset_index()
    )

    fig = px.bar(
        traffic,
        x="Traffic",
        y="Delivery_Time",
        title="Traffic Impact",
    )

    col1.plotly_chart(
        fig,
        use_container_width=True
    )

    # Weather

    weather = (
        orders
        .groupby("Weather")["Delivery_Time"]
        .mean()
        .sort_values()
        .reset_index()
    )

    fig = px.bar(
        weather,
        x="Weather",
        y="Delivery_Time",
        title="Weather Impact",
    )

    col2.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Area

    area = (
        orders
        .groupby("Area")["Delivery_Time"]
        .mean()
        .sort_values()
        .reset_index()
    )

    fig = px.bar(
        area,
        x="Area",
        y="Delivery_Time",
        title="Delivery Time by Area",
    )

    col1.plotly_chart(
        fig,
        use_container_width=True
    )

    # Distance

    fig = px.scatter(
        orders.sample(
            min(10000, len(orders)),
            random_state=42
        ),
        x="Distance_KM",
        y="Delivery_Time",
        color="Traffic",
        title="Distance vs Delivery Time",
        opacity=0.5,
    )

    col2.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🌧️ Monsoon Impact")

    monsoon = (
        orders
        .groupby("Is_Monsoon")["Delivery_Time"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        monsoon,
        x="Is_Monsoon",
        y="Delivery_Time",
        title="Average Delivery Time During Monsoon",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CAPACITY PLANNING
# ============================================================

elif page == "👥 Capacity Planning":

    st.title("👥 Delivery Capacity Planning")

    st.markdown(
        "Recommended delivery capacity based on predicted demand."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Forecast Demand",
        f"{capacity['Forecast_Demand'].mean():.0f}"
    )

    col2.metric(
        "Average Agents Required",
        f"{capacity['Recommended_Agents'].mean():.0f}"
    )

    col3.metric(
        "Peak Agents Required",
        f"{capacity['Recommended_Agents'].max():.0f}"
    )

    st.markdown("---")

    fig = px.line(
        capacity,
        x="Date",
        y=[
            "Forecast_Demand",
            "Planned_Delivery_Capacity"
        ],
        title="Forecast Demand vs Planned Capacity",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.line(
        capacity,
        x="Date",
        y="Recommended_Agents",
        title="Recommended Delivery Agents",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Demand Risk Distribution")

    risk_counts = (
        capacity["Demand_Risk"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Demand_Risk",
        "Days"
    ]

    fig = px.pie(
        risk_counts,
        names="Demand_Risk",
        values="Days",
        title="Demand Risk Distribution",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# OPERATIONAL INTELLIGENCE
# ============================================================

elif page == "⚠️ Operational Intelligence":

    st.title("⚠️ Operational Intelligence Center")

    st.markdown(
        "Daily risk monitoring and operational recommendations."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Risk Score",
        f"{intelligence['Operational_Risk_Score'].mean():.1f}"
    )

    high_risk_days = intelligence[
        intelligence["Operational_Risk_Level"]
        .isin(["High", "Critical"])
    ]

    col2.metric(
        "High Risk Days",
        len(high_risk_days)
    )

    col3.metric(
        "Average Daily Orders",
        f"{intelligence['Total_Orders'].mean():.0f}"
    )

    st.markdown("---")

    # Risk distribution

    risk_distribution = (
        intelligence[
            "Operational_Risk_Level"
        ]
        .value_counts()
        .reset_index()
    )

    risk_distribution.columns = [
        "Risk Level",
        "Days"
    ]

    fig = px.bar(
        risk_distribution,
        x="Risk Level",
        y="Days",
        title="Operational Risk Distribution",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🚨 High-Risk Operational Days")

    display_columns = [
        "Date",
        "Total_Orders",
        "Average_Delivery_Time",
        "Dominant_Traffic",
        "Dominant_Weather",
        "Festival",
        "Operational_Risk_Score",
        "Operational_Risk_Level",
        "Recommended_Action",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in intelligence.columns
    ]

    st.dataframe(
        high_risk_days[
            available_columns
        ].sort_values(
            "Operational_Risk_Score",
            ascending=False
        ),
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Daily Operational Risk Trend")

    fig = px.line(
        intelligence,
        x="Date",
        y="Operational_Risk_Score",
        color="Operational_Risk_Level",
        title="Operational Risk Over Time",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "E-Commerce Delivery Operations Intelligence"
)