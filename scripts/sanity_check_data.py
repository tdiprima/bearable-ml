from itertools import product

import pandas as pd

rows = []

# Generate ranges
heights = range(150, 191, 2)  # 150 → 190
weights = range(45, 96, 3)  # 45 → 95

# Use zip with itertools.product to generate all combinations

for height, weight in product(heights, weights):
    label = int(height >= 175 and weight >= 75)
    rows.append({"height": height, "weight": weight, "label": label})

df = pd.DataFrame(rows)

df_noisy = df.copy()
df_noisy.loc[4, "label"] = 1  # intentional lie

print(df_noisy.head())
print(f"\nTotal rows: {len(df_noisy)}")
print(df_noisy["label"].value_counts())

df_noisy.to_csv("sanity_check.csv")
