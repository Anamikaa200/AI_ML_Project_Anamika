import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="Canada Income Predictor",
    page_icon="📈",
    layout="wide"
)

# ============================
# Header
# ============================
st.markdown("""
<div style='text-align:center;padding:20px;background:linear-gradient(90deg,#0f172a,#1e3a8a);border-radius:12px;'>
<h1 style='color:white;'>🇨🇦 Canada's Capital Yearwise Income Prediction</h1>
<p style='color:#dbeafe;font-size:18px;'>
Predict Canada's Per Capita Income using Machine Learning (Linear Regression)
</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# App Header
st.title("Canada's Capital Yearwise Income Prediction")
st.write("This app uses a simple **Linear Regression** model to predict Canada's per capita income based on historical data.")

# ==========================================
# 1. Load the Dataset
# ==========================================
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "canada_per_capita_income.csv"

df = pd.read_csv(CSV_FILE)

# Show raw data inside an expander
with st.expander("📊 View Historical Data"):
    st.dataframe(df)

# ==========================================
# 2. Prepare Data & Train Model
# ==========================================
X = df[['year']]
y = df['per capita income (US$)']

reg = linear_model.LinearRegression()
reg.fit(X, y)

st.markdown("## 🔮 Predict Income")

left, right = st.columns([1,1])

with left:
    target_year = st.number_input(
        "Select Year",
        min_value=1970,
        max_value=2050,
        value=2020,
        step=1
    )

with right:
    predicted_income = reg.predict([[target_year]])

    st.markdown(f"""
    <div style='background:#e8f5e9;
                padding:25px;
                border-radius:12px;
                text-align:center;
                border-left:8px solid green;'>
        <h3>Predicted Income</h3>
        <h1 style='color:green;'>${predicted_income[0]:,.2f}</h1>
        <p>Per Capita Income in <b>{target_year}</b></p>
    </div>
    """, unsafe_allow_html=True)

# Predict the income
predicted_income = reg.predict([[target_year]])

# Display the result prominently
st.success(f"💰 **Predicted Per Capita Income for {target_year}:** ${predicted_income[0]:,.2f} USD")

# ==========================================
# 4. Double-Checking the Math (Expander)
# ==========================================
with st.expander("🧮 View Model Math Equation (y = mx + b)"):
    m = reg.coef_[0]
    b = reg.intercept_
    st.write(f"**Slope (m):** `{m:.4f}`")
    st.write(f"**Intercept (b):** `{b:.4f}`")
    st.latex(f"\\text{{Income}} = ({m:.2f} \\times {target_year}) + ({b:.2f})")
    st.write(f"**Calculated Result:** ${m * target_year + b:,.2f}")

st.markdown("---")
st.markdown("## 📈 Income Trend Analysis")

chart_col, data_col = st.columns([2,1])

with chart_col:

    fig, ax = plt.subplots(figsize=(10,6))

    ax.scatter(
        df['year'],
        df['per capita income (US$)'],
        color='red',
        marker='+',
        s=60,
        label='Actual Data'
    )

    ax.plot(
        df['year'],
        reg.predict(df[['year']]),
        color='royalblue',
        linewidth=3,
        label='Regression Line'
    )

    ax.scatter(
        target_year,
        predicted_income,
        color='green',
        s=170,
        label='Prediction'
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Income (US$)")
    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)

with data_col:

    st.info("### 📌 Model Summary")

    st.metric("Slope", f"{reg.coef_[0]:.2f}")
    st.metric("Intercept", f"{reg.intercept_:.2f}")
    st.metric("Prediction Year", target_year)


# ==========================================
# 👨‍💻 Developer Profile
# ==========================================

st.markdown("---")

st.markdown("""
<div style='background:#0f172a;
padding:25px;
border-radius:15px;'>

<h2 style='color:white;text-align:center;'>👨‍💻 Developer</h2>

</div>
""", unsafe_allow_html=True)

col1,col2=st.columns([1,4])

with col1:
    st.image(
        "https://avatars.githubusercontent.com/u/9919?s=200&v=4",
        width=140
    )

with col2:

    st.markdown("## **ANAMIKA YADAV**")
    st.write("Machine Learning | Data Science | Python Developer")

    c1,c2=st.columns(2)

    with c1:
        st.link_button(
            "🐙 GitHub",
            "https://github.com/Anamikaa200",
            use_container_width=True
        )

    with c2:
        st.link_button(
            "💼 LinkedIn",
            "www.linkedin.com/in/anamika-yadav-64b688340",
            use_container_width=True
        )

st.markdown("---")
st.caption("© 2026 Anamika Yadav | Built with ❤️ using Streamlit & Scikit-Learn")
with c1:
    st.link_button(
        "🐙 GitHub Profile",
        "https://github.com/Anamikaa200"
    )

with c2:
    st.link_button(
        "💼 LinkedIn Profile",
        "www.linkedin.com/in/anamika-yadav-64b688340"
    )

st.markdown("---")
st.caption("© 2026 Abhay Kumar Gupta | Built with ❤️ using Streamlit & Scikit-Learn")
