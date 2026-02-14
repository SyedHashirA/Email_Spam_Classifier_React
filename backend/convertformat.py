import pandas as pd

df = pd.read_csv("/Users/hashirahmed/Downloads/spam-pdf-classifier-starter/backend/data/newspam.csv")

# Map 1 → spam, 0 → ham
df["label"] = df["label"].map({1: "spam", 0: "ham"})

df.to_csv("/Users/hashirahmed/Downloads/spam-pdf-classifier-starter/backend/data/newspam.csv", index=False)
print("✅ Converted! First few rows:")
print(df.head())