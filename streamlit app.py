import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("qna_data_extracted_2013_2025.csv", sep=";")
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
        ["Number of questions", "Average degree of answering", "Average sentiment"],
        horizontal=True
    )

    # Group by selection
    group_by = st.radio(
        "Group by:",
        ["None", "Theme", "Orientation", "Bank"],
        horizontal=True
    )

    # Filters
    df_plot = df.copy()

    # Filter by Bank
    if "bank" in df.columns:
        selected_banks = st.multiselect("Filter by Bank:", df["bank"].dropna().unique(), default=df["bank"].dropna().unique())
        df_plot = df_plot[df_plot["bank"].isin(selected_banks)]

    # Filter by Theme
    if "theme" in df.columns:
        selected_themes = st.multiselect("Filter by Theme:", df["theme"].dropna().unique(), default=df["theme"].dropna().unique())
        df_plot = df_plot[df_plot["theme"].isin(selected_themes)]

    # Filter by Orientation
    if "orientation" in df.columns:
        selected_orientations = st.multiselect("Filter by Orientation:", df["orientation"].dropna().unique(), default=df["orientation"].dropna().unique())
        df_plot = df_plot[df_plot["orientation"].isin(selected_orientations)]

    # Plotting
    fig, ax = plt.subplots()

    if metric == "Number of questions":
        if group_by == "None":
            counts = df_plot.groupby("period").size().reset_index(name="count")
            ax.plot(counts["period"], counts["count"], marker="o")
            ax.set_ylabel("Number of Questions")
        else:
            counts = df_plot.groupby(["period", group_by.lower()]).size().reset_index(name="count")
            pivot = counts.pivot(index="period", columns=group_by.lower(), values="count").fillna(0)
            if group_by == "Orientation":
                pivot.plot(kind="bar", stacked=True, ax=ax)
            else:
                pivot.plot(kind="line", marker="o", ax=ax)
            ax.set_ylabel("Number of Questions")

    elif metric == "Average degree of answering":
        if group_by == "None":
            avg = df_plot.groupby("period")["answer_score"].mean().reset_index()
            ax.plot(avg["period"], avg["answer_score"], marker="o")
            ax.set_ylabel("Average degree of answering")
            ax.set_ylim(0, 1)
        else:
            avg = df_plot.groupby(["period", group_by.lower()])["answer_score"].mean().reset_index()
            pivot = avg.pivot(index="period", columns=group_by.lower(), values="answer_score").fillna(0)
            pivot.plot(kind="line", marker="o", ax=ax)
            ax.set_ylabel("Average degree of answering")
            ax.set_ylim(0, 1)
    
    elif metric == "Average sentiment":
        if group_by == "None":
            avg = df_plot.groupby("period")["sentiment"].mean().reset_index()
            ax.plot(avg["period"], avg["sentiment"], marker="o")
            ax.set_ylabel("Average Sentiment")
            ax.set_ylim(-1, 1)
        else:
            avg = df_plot.groupby(["period", group_by.lower()])["sentiment"].mean().reset_index()
            pivot = avg.pivot(index="period", columns=group_by.lower(), values="sentiment").fillna(0)
            pivot.plot(kind="line", marker="o", ax=ax)
            ax.set_ylabel("Average Sentiment")
            ax.set_ylim(-1, 1)

    # X-axis formatting
    ax.set_xlabel("Period")
    ax.set_title(f"{metric} over Time" + ("" if group_by == "None" else f" by {group_by}"))
    xticks = [i for i, p in enumerate(df_plot["period"].unique()) if p.endswith("Q1")]
    ax.set_xticks(xticks)
    ax.set_xticklabels([p.split()[0] for p in df_plot["period"].unique()[xticks]], rotation=45)

    st.pyplot(fig)

    st.subheader("Filtered Data Preview")
    st.dataframe(df_plot.head(20))

# Tab 2: RAG Interaction
with tab2:
    st.header("RAG Interaction (Coming Soon)")
    st.info("This tab will allow you to interact with a Retrieval-Augmented Generation system.")
    st.text_area("Ask a question:", height=100)
    st.button("Submit")
    st.write("The RAG interface will display results here.")
