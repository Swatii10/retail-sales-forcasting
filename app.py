"""
STEP 5: Dashboard (Streamlit)

Run after 01_clean_and_eda.py, 02_forecasting.py, 03_recommendation.py have
all been run once (they produce the CSVs this dashboard reads).

"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")
OUT = "outputs"

st.title("📊 Online Store — Customer Behavior & Sales Forecast Dashboard")

df = pd.read_csv(f"{OUT}/cleaned_orders.csv")
sales = df[df["IsCompletedSale"]]

# --- KPI row -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${sales['TotalPrice'].sum():,.0f}")
c2.metric("Completed Orders", f"{sales.shape[0]:,}")
c3.metric("Unique Customers", f"{sales['CustomerID'].nunique():,}")
customers = pd.read_csv(f"{OUT}/customer_level.csv")
c4.metric("Churn Rate (90d)", f"{customers['Churned'].mean():.1%}")

st.divider()

# --- Top products ----------------------------------------------------
st.subheader("🏆 Top-Selling Products")
top_products = pd.read_csv(f"{OUT}/top_products.csv", index_col=0).sort_values(
    "Revenue", ascending=False)
left, right = st.columns([2, 1])
with left:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=top_products["Revenue"], y=top_products.index,
                hue=top_products.index, palette="viridis", legend=False, ax=ax)
    ax.set_xlabel("Revenue ($)")
    st.pyplot(fig)
with right:
    st.dataframe(top_products, use_container_width=True)

st.divider()

# --- Seasonal demand ---------------------------------------------------
st.subheader("📈 Seasonal / Monthly Demand")
monthly = pd.read_csv(f"{OUT}/monthly_sales.csv", index_col=0)
st.line_chart(monthly)

st.divider()

# --- Forecast ------------------------------------------------------
st.subheader("🔮 Next Month's Forecasted Revenue")
forecast = pd.read_csv(f"{OUT}/forecast_results.csv")
fc1, fc2 = st.columns(2)
fc1.metric(f"SMA-3 forecast ({forecast.ForecastMonth[0]})",
           f"${forecast.ForecastedRevenue[0]:,.0f}")
fc2.metric(f"Linear trend forecast ({forecast.ForecastMonth[1]})",
           f"${forecast.ForecastedRevenue[1]:,.0f}")
st.image(f"{OUT}/forecast.png")

st.divider()

# --- Churn -----------------------------------------------------------
st.subheader("👥 Customer Churn")
st.image(f"{OUT}/churn_pie.png", width=400)
st.caption(
    "Churned = no purchase in the last 90 days as of the latest order date "
    "in the dataset. Adjust CHURN_WINDOW_DAYS in 01_clean_and_eda.py to match "
    "your business's real repurchase cycle."
)

st.divider()

# --- Recommendations --------------------------------------------------
st.subheader("🎯 Recommendations")
st.caption(
    "Only 11 of 1,189 customers in this dataset have more than one order, "
    "so true collaborative filtering has very little signal. The table below "
    "is a popularity-based recommender (best for new/anonymous visitors)."
)
popular = pd.read_csv(f"{OUT}/recommend_popular.csv")
st.dataframe(popular, use_container_width=True)
