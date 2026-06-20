"""
STEP 1-3: Data Cleaning, Feature Engineering, and Exploratory Data Analysis
Run this first. It reads the raw Excel export, cleans it, engineers date/customer
features, and produces the EDA charts + aggregated tables used later for forecasting
and recommendations.

Install once: pip install pandas matplotlib seaborn openpyxl
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------
df = pd.read_excel("data_raw.xlsx")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------
# a) Types
df["Date"] = pd.to_datetime(df["Date"])
df["CouponCode"] = df["CouponCode"].fillna("NONE")  # blank = no coupon used

# b) Drop exact duplicate orders, if any
before = len(df)
df = df.drop_duplicates(subset="OrderID")
print(f"Removed {before - len(df)} duplicate OrderIDs")

# c) Sanity-check numeric fields
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (df["TotalPrice"] > 0)]

# d) IMPORTANT: Cancelled/Returned orders are NOT realized revenue.
#    Keep them in the full dataset (useful for churn/ops analysis) but
#    create a "completed" flag so sales/forecasting only uses real revenue.
df["IsCompletedSale"] = df["OrderStatus"].isin(["Delivered", "Shipped"])
print(df["OrderStatus"].value_counts())
print(f"Completed sales rows usable for revenue: {df['IsCompletedSale'].sum()} / {len(df)}")

# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["MonthName"] = df["Date"].dt.strftime("%b")
df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
df["DayOfWeek"] = df["Date"].dt.day_name()


def season(m):
    return {12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"}[m]


df["Season"] = df["Month"].apply(season)

# Simple product -> category mapping (your file has no Category column,
# so we derive one — edit this dict if your real catalog differs)
category_map = {
    "Laptop": "Electronics", "Phone": "Electronics", "Tablet": "Electronics",
    "Monitor": "Electronics", "Printer": "Electronics",
    "Chair": "Furniture", "Desk": "Furniture",
}
df["Category"] = df["Product"].map(category_map)

df.to_csv(f"{OUT}/cleaned_orders.csv", index=False)
print(f"Saved cleaned dataset -> {OUT}/cleaned_orders.csv")

sales = df[df["IsCompletedSale"]].copy()  # use this for every revenue calc below

# ---------------------------------------------------------------
# 4. CUSTOMER-LEVEL TABLE (total spend, frequency, recency)
# ---------------------------------------------------------------
snapshot_date = df["Date"].max() + pd.Timedelta(days=1)
customers = sales.groupby("CustomerID").agg(
    TotalSpend=("TotalPrice", "sum"),
    OrderCount=("OrderID", "nunique"),
    LastPurchase=("Date", "max"),
    FirstPurchase=("Date", "min"),
).reset_index()
customers["DaysSinceLastPurchase"] = (snapshot_date - customers["LastPurchase"]).dt.days
# Flag inactive / churned customers: no purchase in the last 90 days
CHURN_WINDOW_DAYS = 90
customers["Churned"] = customers["DaysSinceLastPurchase"] > CHURN_WINDOW_DAYS
customers.to_csv(f"{OUT}/customer_level.csv", index=False)
churn_rate = customers["Churned"].mean()
print(f"Churn rate (no purchase in {CHURN_WINDOW_DAYS} days): {churn_rate:.1%}")

# ---------------------------------------------------------------
# 5. BEST-SELLING PRODUCTS
# ---------------------------------------------------------------
top_products = (sales.groupby("Product")
                 .agg(Revenue=("TotalPrice", "sum"), UnitsSold=("Quantity", "sum"))
                 .sort_values("Revenue", ascending=False))
top_products.to_csv(f"{OUT}/top_products.csv")
print("\nTop products by revenue:\n", top_products)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_products["Revenue"], y=top_products.index, hue=top_products.index,
            palette="viridis", legend=False)
plt.title("Revenue by Product (completed sales only)")
plt.xlabel("Revenue ($)")
plt.tight_layout()
plt.savefig(f"{OUT}/top_products.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. SEASONAL DEMAND
# ---------------------------------------------------------------
monthly_sales = sales.groupby("YearMonth")["TotalPrice"].sum().sort_index()
monthly_sales.to_csv(f"{OUT}/monthly_sales.csv")

plt.figure(figsize=(11, 5))
monthly_sales.plot(marker="o")
plt.title("Monthly Revenue Trend")
plt.ylabel("Revenue ($)")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/monthly_revenue_trend.png", dpi=150)
plt.close()

season_sales = sales.groupby("Season")["TotalPrice"].sum().reindex(
    ["Winter", "Spring", "Summer", "Fall"])
plt.figure(figsize=(7, 5))
sns.barplot(x=season_sales.index, y=season_sales.values, hue=season_sales.index,
            palette="coolwarm", legend=False)
plt.title("Revenue by Season")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.savefig(f"{OUT}/seasonal_revenue.png", dpi=150)
plt.close()

peak_month = monthly_sales.idxmax()
print(f"\nPeak revenue month: {peak_month} (${monthly_sales.max():,.2f})")

# ---------------------------------------------------------------
# 7. CHURN VISUAL
# ---------------------------------------------------------------
plt.figure(figsize=(6, 5))
customers["Churned"].value_counts().rename({True: "Churned", False: "Active"}).plot(
    kind="pie", autopct="%1.1f%%", colors=["#4CAF50", "#E57373"])
plt.title(f"Customer Status (>{CHURN_WINDOW_DAYS} days inactive = churned)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/churn_pie.png", dpi=150)
plt.close()

print("\nAll EDA charts + tables saved to ./outputs/")
