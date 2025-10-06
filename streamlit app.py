import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("qna_data_both_merged_comma.csv", sep=",")
    df["period"] = df["year"].astype(str) + " Q" + df["quarter"].astype(str)
    return df

df = load_data()

st.title("Quarterly results Q&A Dashboard")

# Tabs
tab1, tab2 = st.tabs(["Plots", "RAG Interaction"])

# Tab 1: Plots
with tab1:
    st.header("Plots Over Time")

    # Metric selection
    metric = st.radio(
        "Select metric:",
        [
            "Number of questions",
            "Average degree of answering",
            "Question sentiment",
            "Answer sentiment",
        ],
        horizontal=True
    )

    # Group by selection
    group_by = st.radio(
        "Group by:",
        ["None", "Theme", "Orientation", "Bank"],
        horizontal=True
    )

    # Chart type selection
    chart_type = st.radio(
        "Chart type:",
        ["Line (absolute)", "Line (percentage)", "Stacked bar (absolute)", "Stacked bar (percentage)"],
        horizontal=True
    )

    # --- FILTERS ---
    st.subheader("Filters")
    df_plot = df.copy()

    def checkbox_filter(df, col_name, label):
        # Filter the DataFrame based on checkbox selections
        if col_name in df.columns:
            # Fill missing values
            df[col_name] = df[col_name].fillna("(missing)")
            unique_vals = sorted(df[col_name].unique())

            with st.expander(f"Filter by {label}", expanded=False):
                selected = []
                cols = st.columns(min(4, len(unique_vals)))  # max 4 columns
                for i, val in enumerate(unique_vals):
                    # default all checked
                    if cols[i % 4].checkbox(str(val), value=True, key=f"{col_name}_{val}_{i}"):
                        selected.append(val)

                # filter dataframe
                df = df[df[col_name].isin(selected)]
        return df

    # Checkbox filters
    df_plot = checkbox_filter(df_plot, "bank", "Bank")
    df_plot = checkbox_filter(df_plot, "theme", "Theme")
    df_plot = checkbox_filter(df_plot, "orientation", "Orientation")
    df_plot = checkbox_filter(df_plot, "question_sentiment", "Question Sentiment")
    df_plot = checkbox_filter(df_plot, "answer_sentiment", "Answer Sentiment")

    # Slider for answer_score
    if "answer_score" in df_plot.columns:
        min_score = float(df_plot["answer_score"].min())
        max_score = float(df_plot["answer_score"].max())
        score_range = st.slider(
            "Filter by Degree of Answering:",
            min_value=0.0,
            max_value=1.0,
            value=(min_score, max_score),
            step=0.05
        )
        df_plot = df_plot[
            (df_plot["answer_score"] >= score_range[0]) &
            (df_plot["answer_score"] <= score_range[1])
        ]

    # Slider for Year
    all_years = sorted(df["year"].unique())
    min_year = int(all_years[0])
    max_year = int(all_years[-1])

    selected_years = st.slider(
        "Select year range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )

    # Filter dataframe based on selected years
    df_plot = df_plot[df_plot["year"].between(selected_years[0], selected_years[1])]

    # Sort by period
    df_plot = df_plot.sort_values(by=["year", "quarter"])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    def plot_chart(pivot, ylabel):
        # Function to plot line or stacked bar charts based on chart_type selection
        ax.clear()  # Clear axes to avoid overlap
        if chart_type == "Line (absolute)":
            pivot.plot(kind="line", marker="o", ax=ax)
            ax.set_ylabel(ylabel)
        elif chart_type == "Line (percentage)":
            pivot_perc = pivot.div(pivot.sum(axis=1), axis=0) * 100
            pivot_perc.plot(kind="line", marker="o", ax=ax)
            ax.set_ylabel(f"{ylabel} (%)")
        elif chart_type == "Stacked bar (absolute)":
            pivot.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel(ylabel)
        elif chart_type == "Stacked bar (percentage)":
            pivot_perc = pivot.div(pivot.sum(axis=1), axis=0) * 100
            pivot_perc.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel(f"{ylabel} (%)")

        # Legend outside
        ax.legend(
            title=pivot.columns.name if pivot.columns.name else "",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

    # Create pivot tables for plotting
    if metric == "Number of questions":
        if group_by == "None":
            counts = df_plot.groupby("period").size().reset_index(name="count")
            pivot = counts.set_index("period")
            plot_chart(pivot, "Number of Questions")
        else:
            counts = df_plot.groupby(["period", group_by.lower()]).size().reset_index(name="count")
            pivot = counts.pivot(index="period", columns=group_by.lower(), values="count").fillna(0)
            plot_chart(pivot, "Number of Questions")

    elif metric == "Average degree of answering":
        if group_by == "None":
            avg = df_plot.groupby("period")["answer_score"].mean().reset_index()
            pivot = avg.set_index("period")
            plot_chart(pivot, "Average degree of answering")
            ax.set_ylim(0, 1)
        else:
            avg = df_plot.groupby(["period", group_by.lower()])["answer_score"].mean().reset_index()
            pivot = avg.pivot(index="period", columns=group_by.lower(), values="answer_score").fillna(0)
            plot_chart(pivot, "Average degree of answering")
            ax.set_ylim(0, 1)

    elif metric in ["Question sentiment", "Answer sentiment"]:
        col = "question_sentiment" if metric == "Question sentiment" else "answer_sentiment"
        if group_by == "None":
            counts = df_plot.groupby(["period", col]).size().reset_index(name="count")
            pivot = counts.pivot(index="period", columns=col, values="count").fillna(0)
            plot_chart(pivot, f"Count of {metric}")
        else:
            counts = df_plot.groupby(["period", group_by.lower(), col]).size().reset_index(name="count")
            pivot = counts.pivot_table(index="period", columns=[group_by.lower(), col], values="count", fill_value=0)
            plot_chart(pivot, f"Count of {metric}")

    # X-axis formatting
    ax.set_xlabel("Period")
    ax.set_title(f"{metric} over Time" + ("" if group_by == "None" else f" by {group_by}"))
    xticks = [i for i, p in enumerate(df_plot["period"].unique()) if p.endswith("Q1")]
    ax.set_xticks(xticks)
    ax.set_xticklabels([p.split()[0] for p in df_plot["period"].unique()[xticks]], rotation=45)

    # Display plot
    st.pyplot(fig)

    # Display filtered extracted questions and answers
    st.markdown("### Filtered Questions and Answers")
    if "extracted_question" in df_plot.columns and "extracted_answer" in df_plot.columns:
        st.dataframe(df_plot[["bank", "year", "quarter", "extracted_question", "extracted_answer"]])
    else:
        st.write("No extracted questions/answers available in the dataset.")


# Tab 2: RAG Interaction
with tab2:
    st.header("RAG Interaction (Coming Soon)")
    st.info("This tab will allow you to interact with a Retrieval-Augmented Generation system.")
    st.text_area("Ask a question:", height=100)
    st.button("Submit")
    st.write("The RAG interface will display results here.")
