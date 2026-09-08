# Task 1: Streamlit-Based Exploratory Data Analysis Interface

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Exploratory Data Analysis Interface")

# 2. Sidebar: Dataset Ingestion
st.sidebar.header("Dataset Ingestion")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read dataset and validate CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Invalid CSV file. Please upload a valid CSV.")
        st.stop()

    # 3. Dataset Overview
    st.subheader("Dataset Overview")

    st.write("**First 5 Rows:**")
    st.dataframe(df.head())

    st.write("**Shape:**")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.write("**Column Data Types:**")
    st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    # Missing value summary
    st.write("**Missing Values per Column:**")
    missing_count = df.isna().sum()
    missing_percentage = (missing_count / len(df) * 100).round(2)

    missing_table = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage
    })
    st.dataframe(missing_table)

    # Basic statistics for numerical columns
    st.write("**Basic Numerical Statistics:**")

    numerical_columns = df.select_dtypes(include="number").columns

    if len(numerical_columns) > 0:
        stats = pd.DataFrame({
            "Mean": df[numerical_columns].mean(),
            "Median": df[numerical_columns].median(),
            "Min": df[numerical_columns].min(),
            "Max": df[numerical_columns].max()
        })
        st.dataframe(stats)
    else:
        st.write("No numerical columns found.")

    # 4. Attribute Selection
    st.sidebar.header("Attribute Selection")

    selected_column = st.sidebar.selectbox(
        "Select an attribute",
        df.columns
    )

    # Detect column type
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    st.write(f"**Selected Attribute:** {selected_column}")
    st.write(f"**Attribute Type:** {column_type}")

    # 5. Visualization Rendering
    st.subheader("Visualization")

    if column_type == "Numerical":
        # Histogram with seaborn
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column].dropna(), kde=False, ax=ax)
        ax.set_title(f"Distribution of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    else:
        # Bar chart for categorical
        counts = df[selected_column].value_counts().head(20)

        fig, ax = plt.subplots()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
        ax.set_title(f"Frequency of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start EDA.")
