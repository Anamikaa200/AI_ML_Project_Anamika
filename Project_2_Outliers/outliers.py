#import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Outlier Detection using Box Plot",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("📊 Outlier Detection using Box Plot")
st.markdown(
"""
This application helps detect **outliers** in any numerical feature
using the **Interquartile Range (IQR)** method and visualizes them
through a **Box Plot**.
"""
)

# -------------------------------------------------------
# File Upload
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully ✅")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_columns) == 0:
        st.error("No numerical columns found in the dataset.")
        st.stop()

    selected_column = st.selectbox(
        "Select Numerical Feature",
        numeric_columns
    )

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    st.subheader("Summary Statistics")

    st.write(df[selected_column].describe())

    # -------------------------------------------------------
    # Box Plot
    # -------------------------------------------------------

    st.subheader("📦 Box Plot (Outlier Visualization)")

    fig, ax = plt.subplots(figsize=(8,2.5))

    ax.boxplot(
        df[selected_column].dropna(),
        vert=False,
        patch_artist=True
    )

    ax.set_xlabel(selected_column)

    st.pyplot(fig)

    # -------------------------------------------------------
    # Histogram
    # -------------------------------------------------------

    st.subheader("Histogram")

    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.hist(df[selected_column].dropna(), bins=30)

    ax2.set_xlabel(selected_column)
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2)

    # -------------------------------------------------------
    # IQR Method
    # -------------------------------------------------------

    Q1 = df[selected_column].quantile(0.25)
    Q3 = df[selected_column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[selected_column] < lower_bound) |
        (df[selected_column] > upper_bound)
    ]

    st.subheader("Outlier Detection Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Lower Bound", round(lower_bound,2))
    col2.metric("Upper Bound", round(upper_bound,2))
    col3.metric("Total Outliers", len(outliers))

    st.subheader("Detected Outliers")

    st.dataframe(outliers)

    # -------------------------------------------------------
    # Clean Dataset
    # -------------------------------------------------------

    cleaned_df = df[
        (df[selected_column] >= lower_bound) &
        (df[selected_column] <= upper_bound)
    ]

    st.subheader("Dataset After Removing Outliers")

    st.write(f"Original Shape : {df.shape}")
    st.write(f"Cleaned Shape : {cleaned_df.shape}")

    csv = cleaned_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Clean Dataset",
        csv,
        "clean_dataset.csv",
        "text/csv"
    )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.markdown("---")

st.markdown("## 👨‍💻 Developed by Abhay Kumar Gupta")

st.markdown(
"""
### 🌐 Connect with Me

🔗 **GitHub:**  
https://github.com/Abhay-cody

💼 **LinkedIn:**  
https://www.linkedin.com/in/abhay-kumar-gupta-104a18397
"""
)
