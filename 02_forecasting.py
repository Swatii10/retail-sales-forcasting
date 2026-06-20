
#STEP 4: Sales Forecasting


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

monthly = pd.read_csv(f"{OUT}/monthly_sales.csv", index_col=0)["TotalPrice"]
monthly.index = pd.PeriodIndex(monthly.index, freq="M")

print("Monthly revenue history:")
print(monthly)

# ---------------------------------------------------------------
# METHOD 1: Simple Moving Average (good baseline, low data needs)
# ---------------------------------------------------------------
WINDOW = 3
sma_forecast = monthly.rolling(WINDOW).mean().iloc[-1]
print(f"\n[SMA-{WINDOW}] Next month forecast: ${sma_forecast:,.2f}")

# ---------------------------------------------------------------
# METHOD 2: Linear trend regression (captures growth/decline direction)
# ---------------------------------------------------------------
X = np.arange(len(monthly)).reshape(-1, 1)
y = monthly.values
lr = LinearRegression().fit(X, y)
next_month_idx = np.array([[len(monthly)]])
lr_forecast = lr.predict(next_month_idx)[0]
print(f"[Linear Trend] Next month forecast: ${lr_forecast:,.2f}")

next_period = monthly.index[-1] + 1
print(f"\nForecast target month: {next_period}")

# ---------------------------------------------------------------
# Plot history + both forecasts
# ---------------------------------------------------------------
plt.figure(figsize=(11, 5))
monthly.plot(marker="o", label="Actual")
plt.scatter([str(next_period)], [sma_forecast], color="orange", s=80,
            label=f"SMA-{WINDOW} forecast", zorder=5)
plt.scatter([str(next_period)], [lr_forecast], color="green", s=80,
            label="Linear trend forecast", zorder=5)
plt.title("Monthly Revenue: Actual vs Next-Month Forecast")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/forecast.png", dpi=150)
plt.close()

pd.DataFrame({
    "Method": ["SMA-3", "LinearTrend"],
    "ForecastMonth": [str(next_period)] * 2,
    "ForecastedRevenue": [sma_forecast, lr_forecast],
}).to_csv(f"{OUT}/forecast_results.csv", index=False)
print(f"\nSaved chart -> {OUT}/forecast.png and table -> {OUT}/forecast_results.csv")

# ---------------------------------------------------------------
