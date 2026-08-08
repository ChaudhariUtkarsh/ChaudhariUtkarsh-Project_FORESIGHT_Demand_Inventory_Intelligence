import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


DATA_PATH  = "data/processed/processed_data.csv"
IMAGE_DIR  = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)


daily = df.groupby("date")["units_sold"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily["date"], daily["units_sold"], color="#2196F3", linewidth=2)
ax.fill_between(daily["date"], daily["units_sold"], alpha=0.15, color="#2196F3")
ax.set_title("Daily Sales Trend", fontsize=15, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "sales_trend.png"), dpi=150)
plt.close()
print("sales_trend.png")


df["month"] = df["date"].dt.to_period("M").astype(str)
monthly = df.groupby("month")["units_sold"].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(monthly["month"], monthly["units_sold"], color="#4CAF50", edgecolor="white")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
ax.set_title("Monthly Sales", fontsize=15, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Units Sold")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "monthly_sales.png"), dpi=150)
plt.close()
print("monthly_sales.png")


cat = df.groupby("category")["units_sold"].sum().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(cat.index, cat.values, color=sns.color_palette("muted", len(cat)))
axes[0].set_title("Units Sold by Category", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Units Sold")
axes[1].pie(cat.values, labels=cat.index, autopct="%1.1f%%",
            colors=sns.color_palette("muted", len(cat)), startangle=140)
axes[1].set_title("Category Share", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "category_sales.png"), dpi=150)
plt.close()
print("category_sales.png")


num_cols = ["units_sold", "unit_price", "list_price",
            "on_hand_units", "on_order_units", "is_weekend"]
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, square=True)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print("correlation_heatmap.png")


model_path = os.path.join("models", "best_model.pkl")
if os.path.exists(model_path):
    import joblib
    model = joblib.load(model_path)
    if hasattr(model, "feature_importances_"):
        feat_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"f{i}" for i in range(len(model.feature_importances_))]
        fi = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(9, 6))
        fi.plot(kind="barh", ax=ax, color="#FF7043")
        ax.set_title("Feature Importance (Best Model)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, "feature_importance.png"), dpi=150)
        plt.close()
        print("feature_importance.png  (from trained model)")
    else:
        raise AttributeError("Model has no feature_importances_")
else:
    num_feats = ["unit_price", "list_price", "on_hand_units",
                 "on_order_units", "is_weekend"]
    num_feats = [c for c in num_feats if c in df.columns]
    importance = df[num_feats].corrwith(df["units_sold"]).abs().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    importance.plot(kind="barh", ax=ax, color="#FF7043")
    ax.set_title("Feature Importance (Correlation Proxy)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Absolute Correlation with Units Sold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("feature_importance.png (correlation proxy — run train_model.py for real importances)")


df_risk = df.copy()
df_risk["forecast_demand"] = (df_risk.groupby("sku_id")["units_sold"].transform(lambda x: x.rolling(7, min_periods=1).mean()))
df_risk["stockout_score"]  = (df_risk["forecast_demand"] - df_risk["on_hand_units"]).clip(lower=0)
df_risk["overstock_score"] = (df_risk["on_hand_units"] - df_risk["forecast_demand"]).clip(lower=0)
max_score = max(df_risk[["stockout_score", "overstock_score"]].max().max(), 1)
df_risk["risk_score"] = (df_risk[["stockout_score", "overstock_score"]].max(axis=1) / max_score * 100).round(2)
df_risk["risk_level"] = df_risk["risk_score"].apply(lambda s: "High" if s >= 70 else ("Medium" if s >= 40 else "Low"))

risk_counts = df_risk["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
colors = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df_risk["risk_score"], bins=20, color="#9C27B0", edgecolor="white")
axes[0].set_title("Risk Score Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Risk Score")
axes[0].set_ylabel("Frequency")

axes[1].bar(risk_counts.index, risk_counts.values, color=[colors[l] for l in risk_counts.index], edgecolor="white")
for i, v in enumerate(risk_counts.values):
    axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=10)
axes[1].set_title("Risk Level Counts", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Risk Level")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "risk_distribution.png"), dpi=150)
plt.close()
print("risk_distribution.png")

print("\nAll images saved to images/")