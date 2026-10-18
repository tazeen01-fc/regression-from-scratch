import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/data.csv")

features = ["hp", "attack", "defense", "sp_attack", "sp_defense",
            "speed", "height_m", "weight_kg", "capture_rate", "base_happiness"]
target = "base_experience"

# 1. Summary of the target
print("Target summary:")
print(df[target].describe())

# 2. How strongly does each feature relate to the target?
print("\nCorrelation with", target)
print(df[features + [target]].corr()[target].sort_values(ascending=False))

# 3. The range of each feature (this shows why scaling matters)
print("\nFeature ranges:")
print(df[features].describe().T[["min", "max", "mean"]])

# 4. Histogram of the target
plt.hist(df[target], bins=30)
plt.title("Distribution of base_experience")
plt.xlabel("base_experience")
plt.ylabel("count")
plt.show()