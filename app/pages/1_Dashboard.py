import os
import pandas as pd
import streamlit as st


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

RISK_FILE = os.path.join(PROCESSED_DIR, "inventory_risk_scores.csv")
REORDER_FILE = os.path.join(PROCESSED_DIR,"reorder_priority_list.csv")
MARKDOWN_FILE = os.path.join(PROCESSED_DIR, "markdown_clear_priority_list.csv")
WEEKLY_FILE = os.path.join(PROCESSED_DIR, "weekly_model_data.csv")


st.set_page_config(page_title="Project Foresight", page_icon=" ", layout="wide")
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }
    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
    }
    .metric-title {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">Project Foresight</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">' 'Inventory Risk, Reorder & Markdown/Clear ' 'Decision Intelligence Dashboard' '</div>', unsafe_allow_html=True)


st.subheader("Current Model Performance")
st.caption("Evaluation Method: Rolling-Origin Cross-Validation")
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Baseline WAPE", "12.17%")

with m2:
    st.metric("LightGBM WAPE", "8.16%")

with m3:
    st.metric("Improvement", "33.01%")

st.info("Production evaluation is based on a 52-week Seasonal Naive " "baseline and Rolling-Origin Cross-Validation using WAPE.")
st.divider()


@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Error reading file: {os.path.basename(path)}")
        st.error(str(e))
        return None

    df.columns = (df.columns.astype(str).str.strip())

    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

    if "sku_id" in df.columns:
        df["sku_id"] = (df["sku_id"].astype(str).str.strip())

    if "category" in df.columns:
        df["category"] = (df["category"].astype(str).str.strip())

    if "risk_level" in df.columns:
        df["risk_level"] = (df["risk_level"].astype(str).str.strip().str.upper())

    if "reorder_priority" in df.columns:
        df["reorder_priority"] = (df["reorder_priority"].astype(str).str.strip().str.upper())

    if "markdown_clear_priority" in df.columns:
        df["markdown_clear_priority"] = (df["markdown_clear_priority"].astype(str).str.strip().str.upper())
    return df

risk_df = load_csv(RISK_FILE)
reorder_df = load_csv(REORDER_FILE)
markdown_df = load_csv(MARKDOWN_FILE)
weekly_df = load_csv(WEEKLY_FILE)


if risk_df is None:
    st.error("inventory_risk_scores.csv not found.")
    st.info(
        "Run these commands first:\n\n"
        "python src/pipeline.py\n\n"
        "python src/train_model.py\n\n"
        "python src/risk_scoring.py"
    )
    st.stop()


if "category" not in risk_df.columns:
    if (weekly_df is not None and "sku_id" in weekly_df.columns and "category" in weekly_df.columns):
        category_lookup = (weekly_df[["sku_id", "category"]].dropna(subset=["sku_id"]).copy())
        category_lookup["sku_id"] = (category_lookup["sku_id"].astype(str).str.strip())
        category_lookup["category"] = (category_lookup["category"].astype(str).str.strip())
        category_lookup = (category_lookup.drop_duplicates(subset=["sku_id"]))
        risk_df = risk_df.merge(category_lookup, on="sku_id", how="left")


if (reorder_df is not None and "category" not in reorder_df.columns and weekly_df is not None and "sku_id" in reorder_df.columns and "sku_id" in weekly_df.columns and "category" in weekly_df.columns):
    category_lookup = (weekly_df[["sku_id", "category"]].dropna(subset=["sku_id"]).copy())
    category_lookup["sku_id"] = (category_lookup["sku_id"].astype(str).str.strip())
    category_lookup["category"] = (category_lookup["category"].astype(str).str.strip())
    category_lookup = (category_lookup.drop_duplicates(subset=["sku_id"]))
    reorder_df = reorder_df.merge(category_lookup, on="sku_id", how="left")

if (markdown_df is not None and "category" not in markdown_df.columns and weekly_df is not None and "sku_id" in markdown_df.columns and "sku_id" in weekly_df.columns and "category" in weekly_df.columns):
    category_lookup = (weekly_df[["sku_id", "category"]].dropna(subset=["sku_id"]).copy())
    category_lookup["sku_id"] = (category_lookup["sku_id"].astype(str).str.strip())
    category_lookup["category"] = (category_lookup["category"].astype(str).str.strip())
    category_lookup = (category_lookup.drop_duplicates(subset=["sku_id"]))
    markdown_df = markdown_df.merge(category_lookup, on="sku_id", how="left")


st.sidebar.title("Dashboard Filters")
category_options = ["ALL"]
if "category" in risk_df.columns:
    available_categories = (risk_df["category"].dropna().astype(str).str.strip().unique().tolist())
    available_categories = [x for x in available_categories if x and x.lower() != "nan"]
    category_options += sorted(available_categories)
selected_category = st.sidebar.selectbox("Category", category_options)


sku_options = ["ALL"]
if "sku_id" in risk_df.columns:
    sku_values = (risk_df["sku_id"].dropna().astype(str).str.strip().unique().tolist())
    sku_options += sorted(sku_values)
selected_sku = st.sidebar.selectbox("SKU", sku_options)


risk_levels = ["ALL"]
if "risk_level" in risk_df.columns:
    available_risks = (risk_df["risk_level"].dropna().astype(str).str.strip().str.upper().unique().tolist())
    preferred_order = ["HIGH", "MEDIUM", "LOW"]
    available_risks = [ x for x in preferred_order if x in available_risks]
    risk_levels += available_risks
selected_risk = st.sidebar.selectbox("Risk Level", risk_levels)


reorder_priority_options = ["ALL"]
if (reorder_df is not None and "reorder_priority" in reorder_df.columns):
    priorities = (reorder_df["reorder_priority"].dropna().astype(str).str.strip().str.upper().unique().tolist())
    priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    priorities = [x for x in priority_order if x in priorities]
    reorder_priority_options += priorities
selected_reorder_priority = st.sidebar.selectbox("Reorder Priority", reorder_priority_options)


markdown_priority_options = ["ALL"]
if (markdown_df is not None and "markdown_clear_priority" in markdown_df.columns):
    priorities = (markdown_df["markdown_clear_priority"].dropna().astype(str).str.strip().str.upper().unique().tolist())
    priority_order = ["HIGH", "MEDIUM", "LOW"]
    priorities = [x for x in priority_order if x in priorities]
    markdown_priority_options += priorities
selected_markdown_priority = st.sidebar.selectbox("Markdown/Clear Priority", markdown_priority_options)


filtered_risk = risk_df.copy()
if (selected_category != "ALL" and "category" in filtered_risk.columns):
    filtered_risk = filtered_risk[filtered_risk["category"].astype(str).str.strip() == selected_category].copy()

if (selected_sku != "ALL" and "sku_id" in filtered_risk.columns):
    filtered_risk = filtered_risk[filtered_risk["sku_id"].astype(str).str.strip() == str(selected_sku)].copy()

if (selected_risk != "ALL" and "risk_level" in filtered_risk.columns):
    filtered_risk = filtered_risk[filtered_risk["risk_level"] == selected_risk].copy()

if reorder_df is not None:
    filtered_reorder = reorder_df.copy()
    if (selected_category != "ALL" and "category" in filtered_reorder.columns):
        filtered_reorder = filtered_reorder[filtered_reorder["category"].astype(str).str.strip() == selected_category].copy()
    if (selected_sku != "ALL" and "sku_id" in filtered_reorder.columns):
        filtered_reorder = filtered_reorder[filtered_reorder["sku_id"].astype(str).str.strip() == str(selected_sku)].copy()
    if (selected_reorder_priority != "ALL" and "reorder_priority" in filtered_reorder.columns):
        filtered_reorder = filtered_reorder[filtered_reorder["reorder_priority"] == selected_reorder_priority].copy()
else:
    filtered_reorder = None


if markdown_df is not None:
    filtered_markdown = markdown_df.copy()
    if (selected_category != "ALL" and "category" in filtered_markdown.columns):
        filtered_markdown = filtered_markdown[filtered_markdown["category"].astype(str).str.strip() == selected_category].copy()
    if (selected_sku != "ALL" and "sku_id" in filtered_markdown.columns):
        filtered_markdown = filtered_markdown[filtered_markdown["sku_id"].astype(str).str.strip() == str(selected_sku)].copy()
    if (selected_markdown_priority != "ALL" and "markdown_clear_priority" in filtered_markdown.columns):
        filtered_markdown = filtered_markdown[filtered_markdown["markdown_clear_priority"] == selected_markdown_priority].copy()
else:
    filtered_markdown = None


st.sidebar.markdown("---")
st.sidebar.write("### Selected Filters")
st.sidebar.write(f"**Category:** {selected_category}")
st.sidebar.write(f"**SKU:** {selected_sku}")
st.sidebar.write(f"**Risk Level:** {selected_risk}")
st.sidebar.write(f"**Reorder Priority:** " f"{selected_reorder_priority}")
st.sidebar.write(f"**Markdown/Clear Priority:** " f"{selected_markdown_priority}")
st.sidebar.markdown("---")
st.sidebar.metric("Filtered Risk Records", f"{len(filtered_risk):,}")


st.subheader("Inventory Risk Overview")
total_records = len(filtered_risk)

if "risk_level" in filtered_risk.columns:
    high_risk_count = (filtered_risk["risk_level"].eq("HIGH").sum())
    medium_risk_count = (filtered_risk["risk_level"].eq("MEDIUM").sum())
    low_risk_count = (filtered_risk["risk_level"].eq("LOW").sum())
else:
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0

if "risk_score" in filtered_risk.columns:
    risk_values = pd.to_numeric(filtered_risk["risk_score"], errors="coerce")
    avg_risk = risk_values.mean()
    if pd.isna(avg_risk):
        avg_risk = 0
else:
    avg_risk = 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Records", f"{total_records:,}")

with col2:
    st.metric("High Risk", f"{high_risk_count:,}")

with col3:
    st.metric("Medium Risk", f"{medium_risk_count:,}")

with col4:
    st.metric("Low Risk", f"{low_risk_count:,}")

with col5:
    st.metric("Average Risk", f"{avg_risk:.2f}")


if (medium_risk_count == 0 and total_records > 0):
    st.warning("No MEDIUM risk records found in the " "current filtered dataset.")
st.divider()


st.subheader("Reorder Decision Summary")
if filtered_reorder is not None:
    reorder_count = len(filtered_reorder)
    reorder_units = 0
    possible_reorder_columns = ["reorder_units", "recommended_reorder_units", "recommended_reorder_qty", "reorder_quantity"]

    for column in possible_reorder_columns:
        if column in filtered_reorder.columns:
            values = pd.to_numeric(filtered_reorder[column], errors="coerce").fillna(0)
            reorder_units = values.sum()
            break

    if "reorder_priority" in filtered_reorder.columns:
        critical_reorders = (filtered_reorder["reorder_priority"].eq("CRITICAL").sum())
        high_reorders = (filtered_reorder["reorder_priority"].eq("HIGH").sum())
        medium_reorders = (filtered_reorder["reorder_priority"].eq("MEDIUM").sum())
        low_reorders = (filtered_reorder["reorder_priority"].eq("LOW").sum())
    else:
        critical_reorders = 0
        high_reorders = 0
        medium_reorders = 0
        low_reorders = 0

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric("Total Reorder Records", f"{reorder_count:,}")

    with r2:
        st.metric("Critical", f"{critical_reorders:,}")

    with r3:
        st.metric("Medium", f"{medium_reorders:,}")

    with r4:
        st.metric("Reorder Units", f"{reorder_units:,.0f}")

    if (reorder_count > 0 and reorder_units == 0):
        st.warning("Reorder records exist, but total " "Reorder Units are 0. Check the " "reorder quantity column.")

else:
    st.warning("reorder_priority_list.csv not found.")


st.subheader("Prioritised Reorder List")
if (filtered_reorder is not None and len(filtered_reorder) > 0):
    display_reorder = (filtered_reorder.copy())

    if "reorder_priority" in display_reorder.columns:
        priority_order = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
        display_reorder["_priority_order"] = (display_reorder["reorder_priority"].map(priority_order).fillna(99))
        display_reorder = (display_reorder.sort_values("_priority_order").drop(columns="_priority_order"))
    st.dataframe(display_reorder, use_container_width=True, hide_index=True)
    reorder_csv = (display_reorder.to_csv(index=False).encode("utf-8"))
    st.download_button(label="Download Reorder Priority List", data=reorder_csv, file_name="reorder_priority_list.csv", mime="text/csv")
else:
    st.info("No reorder records found " "for current filters.")
st.divider()

st.subheader("Markdown / Clear Decision Summary")

if filtered_markdown is not None:
    markdown_count = len(filtered_markdown)
    excess_units = 0
    possible_excess_columns = ["excess_inventory_units", "excess_units", "excess_inventory", "excess_quantity"]

    for column in possible_excess_columns:
        if column in filtered_markdown.columns:
            values = pd.to_numeric(filtered_markdown[column], errors="coerce").fillna(0)
            excess_units = values.sum()
            break


    if ("markdown_clear_priority" in filtered_markdown.columns):
        high_markdown = (filtered_markdown["markdown_clear_priority"].eq("HIGH").sum())
        medium_markdown = (filtered_markdown["markdown_clear_priority"].eq("MEDIUM").sum())
        low_markdown = (filtered_markdown["markdown_clear_priority"].eq("LOW").sum())
    else:
        high_markdown = 0
        medium_markdown = 0
        low_markdown = 0


    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("Total Markdown/Clear", f"{markdown_count:,}")

    with m2:
        st.metric("High", f"{high_markdown:,}")

    with m3:
        st.metric("Medium", f"{medium_markdown:,}")

    with m4:
        st.metric("Low", f"{low_markdown:,}")

    with m5:
        st.metric("Excess Inventory", f"{excess_units:,.0f}")

else:
    st.warning("markdown_clear_priority_list.csv " "not found.")


st.subheader("Prioritised Markdown / Clear List")

if (filtered_markdown is not None and len(filtered_markdown) > 0):
    display_markdown = (filtered_markdown.copy())

    if ("markdown_clear_priority" in display_markdown.columns):
        priority_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}

        display_markdown["_priority_order"] = (display_markdown["markdown_clear_priority"].map(priority_order).fillna(99))
        display_markdown = (display_markdown.sort_values("_priority_order").drop(columns="_priority_order"))

    st.dataframe(display_markdown, use_container_width=True, hide_index=True)
    markdown_csv = (display_markdown.to_csv(index=False).encode("utf-8"))
    st.download_button(label="Download Markdown/Clear Priority List", data=markdown_csv, file_name="markdown_clear_priority_list.csv", mime="text/csv")

else:
    st.info("No markdown/clear records found " "for current filters.")
st.divider()


st.subheader("Decision Intelligence")
d1, d2 = st.columns(2)

with d1:
    st.markdown("### Reorder Action")

    if (filtered_reorder is not None and len(filtered_reorder) > 0):
        if ("reorder_priority" in filtered_reorder.columns):
            critical = (filtered_reorder["reorder_priority"].eq("CRITICAL").sum())
            high = (filtered_reorder["reorder_priority"].eq("HIGH").sum())
            medium = (filtered_reorder["reorder_priority"].eq("MEDIUM").sum())
            low = (filtered_reorder["reorder_priority"].eq("LOW").sum())

            st.write(f"**Critical Reorders:** " f"{critical:,}")
            st.write(f"**High Reorders:** " f"{high:,}")
            st.write(f"**Medium Reorders:** " f"{medium:,}")
            st.write(f"**Low Reorders:** " f"{low:,}")

            if critical > 0:
                st.error("Immediate supplier/" "replenishment action required.")
            elif high > 0:
                st.error("High-priority replenishment " "action required.")
            elif medium > 0:
                st.warning("Plan replenishment for " "medium-priority SKUs.")
            else:
                st.success("No critical reorder " "action required.")

    else:
        st.success("No reorder action required " "for current filter.")

with d2:
    st.markdown("### Markdown / Clear Action")

    if (filtered_markdown is not None and len(filtered_markdown) > 0):
        if ("markdown_clear_priority" in filtered_markdown.columns):
            high = (filtered_markdown["markdown_clear_priority"].eq("HIGH").sum())
            medium = (filtered_markdown["markdown_clear_priority"].eq("MEDIUM").sum())
            low = (filtered_markdown["markdown_clear_priority"].eq("LOW").sum())

            st.write(f"**High Markdown/Clear:** " f"{high:,}")
            st.write(f"**Medium Markdown/Clear:** " f"{medium:,}")
            st.write(f"**Low Markdown/Clear:** " f"{low:,}")

            if high > 0:
                st.error("Prioritise these SKUs " "for markdown/clearance.")
            elif medium > 0:
                st.warning("Consider promotional pricing " "for medium-priority inventory.")
            elif low > 0:
                st.info("Low-priority markdown/clear " "actions identified.")
            else:
                st.success("No urgent markdown/clear " "action required.")
    else:
        st.success("No markdown/clear action required " "for current filter.")
st.divider()


st.subheader("Risk Distribution")

if ("risk_level" in filtered_risk.columns and len(filtered_risk) > 0):
    risk_distribution = (filtered_risk["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0).rename_axis("Risk Level").reset_index(name="Records"))
    st.bar_chart(risk_distribution.set_index("Risk Level"))

else:
    st.info("No risk distribution data available.")


st.subheader("Top High-Risk Inventory")

if ("risk_score" in filtered_risk.columns and len(filtered_risk) > 0):
    filtered_risk = (filtered_risk.copy())
    filtered_risk["risk_score"] = (pd.to_numeric(filtered_risk["risk_score"], errors="coerce"))

    top_risk = (filtered_risk.sort_values("risk_score", ascending=False).head(20))
    top_columns = [
        "sku_id",
        "category",
        "week_start",
        "forecast_weekly_demand",
        "on_hand_units",
        "on_order_units",
        "available_inventory",
        "projected_inventory",
        "inventory_coverage_weeks",
        "risk_score",
        "risk_level",
        "stockout_risk",
        "risk_reason",
        "recommended_action"
    ]
    top_columns = [col for col in top_columns if col in top_risk.columns]

    if len(top_columns) > 0:
        st.dataframe(top_risk[top_columns], use_container_width=True, hide_index=True)
    else:
        st.dataframe(top_risk, use_container_width=True, hide_index=True)

else:
    st.info("No high-risk inventory data available.")


with st.expander("Data Diagnostics"):
    st.write("### Risk Dataset")
    st.write(f"Rows: {len(risk_df):,}")
    st.write("Columns:", risk_df.columns.tolist())

    if "category" in risk_df.columns:
        st.write("### Category Distribution")
        category_distribution = (risk_df["category"].value_counts().rename_axis("Category").reset_index(name="Records"))
        st.dataframe(category_distribution, hide_index=True)

    if "risk_level" in risk_df.columns:
        st.write("### Risk Level Distribution")
        risk_distribution = (risk_df["risk_level"].value_counts().rename_axis("Risk Level").reset_index(name="Records"))
        st.dataframe(risk_distribution, hide_index=True)

    if reorder_df is not None:
        st.write("### Reorder Dataset")
        st.write(f"Rows: {len(reorder_df):,}")
        st.write("Columns:", reorder_df.columns.tolist())

        reorder_quantity_columns = [col for col in ["reorder_units", "recommended_reorder_units", "recommended_reorder_qty", "reorder_quantity"] if col in reorder_df.columns]
        st.write("Reorder quantity columns found:", reorder_quantity_columns)

        if reorder_quantity_columns:
            for col in reorder_quantity_columns:
                total = (pd.to_numeric(reorder_df[col], errors="coerce").fillna(0).sum())
                st.write(f"{col}: {total:,.0f}")


    if markdown_df is not None:
        st.write("### Markdown Dataset")
        st.write(f"Rows: {len(markdown_df):,}")
        st.write("Columns:", markdown_df.columns.tolist())

    if weekly_df is not None:
        st.write("### Weekly Model Dataset")
        st.write(f"Rows: {len(weekly_df):,}")
        st.write("Columns:", weekly_df.columns.tolist())

        if "category" in weekly_df.columns:
            st.write("Category column available in " "weekly_model_data.csv.")

        else:
            st.warning("Category column is not available " "in weekly_model_data.csv.")


st.divider()
st.caption("Project Foresight | Inventory Risk, Reorder & " "Markdown/Clear Decision Intelligence")