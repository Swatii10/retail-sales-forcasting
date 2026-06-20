
#STEP 4b: Recommendation System (Optional)


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

OUT = "outputs"
df = pd.read_csv(f"{OUT}/cleaned_orders.csv")
sales = df[df["IsCompletedSale"]]

# ---------------------------------------------------------------
# 1. POPULARITY-BASED ("Best sellers / Trending")
# ---------------------------------------------------------------
popularity = (sales.groupby("Product")["TotalPrice"]
              .sum().sort_values(ascending=False))
print("Top recommendations for a new/anonymous customer (best sellers):")
print(popularity.head(5))
popularity.to_csv(f"{OUT}/recommend_popular.csv")

# ---------------------------------------------------------------
# 2. CO-PURCHASE SIMILARITY
#    Build a Customer x Product matrix, then compute product-product
#    cosine similarity. This tells you "Phone buyers also tend to buy X".
# ---------------------------------------------------------------
basket = pd.crosstab(sales["CustomerID"], sales["Product"])
sim = cosine_similarity(basket.T)
sim_df = pd.DataFrame(sim, index=basket.columns, columns=basket.columns)
sim_df.to_csv(f"{OUT}/product_similarity.csv")


def recommend_similar(product, top_n=3):
    if product not in sim_df:
        return []
    scores = sim_df[product].drop(product).sort_values(ascending=False)
    return list(scores.head(top_n).index)


print("\nExample: 'customers who bought X also liked...'")
for p in basket.columns:
    print(f"  {p:8s} -> {recommend_similar(p)}")

# ---------------------------------------------------------------
