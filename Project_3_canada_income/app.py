import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Set up page configuration
st.set_page_config(page_title="Canada Income Predictor", layout="centered")

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

# ==========================================
# 3. Interactive User Input & Prediction
# ==========================================
st.subheader("🔮 Make a Prediction")
# Add an input for the user to select the year
target_year = st.number_input("Enter or select a year to predict:", min_value=1970, max_value=2050, value=2020, step=1)

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

# ==========================================
# 5. Full Visualisation Plot
# ==========================================
st.subheader("📈 Visualization & Regression Line")

fig, ax = plt.subplots(figsize=(10, 6))
# Plot actual data points
ax.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+', label='Actual Data', s=60)
# Plot the trendline
ax.plot(df['year'], reg.predict(df[['year']]), color='blue', linewidth=2, label='Regression Line')
# Highlight the user's predicted point
ax.scatter(target_year, predicted_income, color='green', marker='o', s=150, label=f'Prediction ({target_year})', zorder=5)

# Graph styling
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Per Capita Income (US$)', fontsize=12)
ax.set_title("Canada's Per Capita Income Trend Over Time", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=11)

# Pass the matplotlib figure to streamlit
st.pyplot(fig)

# ==========================================
# 👨‍💻 Developer Profile
# ==========================================

st.markdown("---")
st.markdown("## 👨‍💻 Developer")

col1, col2 = st.columns([1, 3])

with col1:
    st.image(
        "https://avatars.githubusercontent.com/u/9919?s=200&v=4",
        width=120
    )

with col2:
    st.markdown("### **ABHAY KUMAR GUPTA**")
    st.markdown("**Machine Learning | Data Science | Python Developer**")

st.markdown("### 🔗 Connect With Me")

c1, c2 = st.columns(2)

with c1:
    st.link_button(
        "🐙 GitHub Profile",
        "https://github.com/Abhay-cody"
    )

with c2:
    st.link_button(
        "💼 LinkedIn Profile",
        "https://www.linkedin.com/in/abhay-kumar-gupta-104a18397"
    )

st.markdown("---")
st.caption("© 2026 Abhay Kumar Gupta | Built with ❤️ using Streamlit & Scikit-Learn")
