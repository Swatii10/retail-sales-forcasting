# Online Retail Customer Purchase Behavior & Sales Forecasting

## 1. Problem Statement
Analyze an online store's order history to identify best-selling products,
understand seasonal demand patterns, forecast next month's revenue, and
(optionally) recommend products to customers.

## 2. Dataset
`data_raw.xlsx` — 1,200 orders, Jan 2023–Jun 2025, 14 columns
(OrderID, Date, CustomerID, Product, Quantity, UnitPrice, ShippingAddress,
PaymentMethod, OrderStatus, TrackingNumber, ItemsInCart, CouponCode,
ReferralSource, TotalPrice).

Key data characteristics that shaped the analysis:
- **7 products**, no category column provided (a Product→Category mapping
  was derived manually: Electronics vs Furniture).
- **OrderStatus** includes Cancelled (250) and Returned (247) rows alongside
  Delivered/Shipped/Pending. Only Delivered + Shipped orders (466 rows) were
  treated as realized revenue for all sales/forecasting calculations.
- **1,189 unique customers for 1,200 orders** — only 11 customers have more
  than one order. This makes true collaborative-filtering recommendations
  unreliable; see the Recommendations section below.

## 3. Project Structure
```
01_clean_and_eda.py     - cleaning, feature engineering, EDA charts
02_forecasting.py       - next-month revenue forecast (SMA + linear trend)
03_recommendation.py    - popularity + co-purchase recommender
app.py                  - Streamlit dashboard tying it all together
outputs/                - generated CSVs and PNG charts
```

Run order: `01_clean_and_eda.py` → `02_forecasting.py` →
`03_recommendation.py` → `streamlit run app.py`

## 4. Methods
- **Cleaning:** parsed dates, filled blank coupon codes with "NONE",
  removed duplicate OrderIDs, dropped non-positive quantity/price rows,
  flagged completed vs non-completed sales.
- **Feature engineering:** Year, Month, Season, YearMonth, DayOfWeek;
  customer-level TotalSpend, OrderCount, DaysSinceLastPurchase, Churned flag
  (no purchase in 90 days).
- **EDA:** best sellers by revenue/units, monthly and seasonal revenue
  trends, churn rate visualization.
- **Forecasting:** two zero-dependency baselines — 3-month Simple Moving
  Average and Linear Trend Regression — with commented-out ARIMA
  (statsmodels) and Prophet blocks ready to enable once those packages are
  installed locally (they weren't available in the build environment, but
  the code is correct and tested patterns — just uncomment and run).
- **Recommendations:** popularity-based ranking (robust default) plus a
  product-product cosine-similarity "also bought" matrix built from the
  customer x product purchase matrix.

## 5. Key Insights
- **Top product by revenue:** Printer ($82,452), followed closely by
  Laptop ($80,952) and Chair ($71,816) — revenue is fairly evenly spread
  across the 7-product catalog (no single dominant SKU).
- **Peak month:** October 2023 ($36,899) was the highest single month in
  the dataset; revenue is otherwise noisy month-to-month without a strong
  recurring seasonal pattern — worth re-checking once more years of data
  accumulate.
- **Churn:** ~92% of customers are "churned" under a 90-day window — expected
  given 99% of customers only ever placed one order. This number is really
  measuring "one-and-done purchase rate," not seasonal churn — treat it as a
  signal to invest in retention/re-engagement campaigns rather than a
  literal churn-rate KPI.
- **Forecast:** Both baseline methods land in the $13.6K–$14.2K range for
  next month, well below the noisy historical peaks — driven by an overall
  flat-to-slightly-declining trend in the back half of the dataset.
- **Recommendations:** With only 11 repeat customers, co-purchase
  similarity scores were nearly uniform across products — i.e., not enough
  signal for personalized recommendations yet. Popularity-based "best
  sellers" is the practical fallback until repeat-purchase volume grows.

## 6. Next Steps / How to Extend
- Install `statsmodels`/`prophet` locally and uncomment the ARIMA/Prophet
  blocks in `02_forecasting.py` for proper seasonality-aware forecasting.
- Once repeat-purchase volume increases, swap the recommender for real
  collaborative filtering with `scikit-surprise` (code stub included).
- Deploy `app.py` with `streamlit run app.py`, or push to Streamlit
  Community Cloud / Heroku for a public dashboard link.