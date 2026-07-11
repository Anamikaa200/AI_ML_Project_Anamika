#Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Clustering",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌸 Iris Flower Clustering using K-Means")
st.markdown(
    """
This project demonstrates **Unsupervised Machine Learning** using the famous
**Iris Dataset**. The clustering is performed using only:

- 🌼 Petal Length
- 🌼 Petal Width

The project also demonstrates:

- ✅ K-Means Clustering
- ✅ Feature Scaling (MinMaxScaler)
- ✅ Elbow Method
"""
)

st.divider()

# -----------------------------
# Load Dataset
# -----------------------------
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df = df[['petal length (cm)', 'petal width (cm)']]

# -----------------------------
# Dataset Preview
# -----------------------------
st.header("📊 Dataset Preview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Samples", len(df))

with col2:
    st.metric("Features Used", 2)

st.dataframe(df, use_container_width=True)

st.divider()

# -----------------------------
# Scatter Plot
# -----------------------------
st.header("🌼 Original Dataset Visualization")

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(
    df['petal length (cm)'],
    df['petal width (cm)'],
    color='purple'
)

ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")
ax.set_title("Petal Length vs Petal Width")

st.pyplot(fig)

st.divider()

# -----------------------------
# K Value
# -----------------------------
st.header("🎯 K-Means Clustering")

k = st.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3
)

# -----------------------------
# Without Scaling
# -----------------------------
km = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df['Cluster'] = km.fit_predict(
    df[['petal length (cm)', 'petal width (cm)']]
)

fig2, ax2 = plt.subplots(figsize=(7,6))

scatter = ax2.scatter(
    df['petal length (cm)'],
    df['petal width (cm)'],
    c=df['Cluster'],
    cmap='viridis',
    s=70
)

ax2.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    color='red',
    marker='*',
    s=300,
    label='Centroids'
)

ax2.set_xlabel("Petal Length")
ax2.set_ylabel("Petal Width")
ax2.set_title("K-Means Clustering")
ax2.legend()

st.pyplot(fig2)

st.divider()

# -----------------------------
# Scaling
# -----------------------------
st.header("📈 Feature Scaling")

st.write(
    "MinMaxScaler scales both features into the range **0 to 1**, "
    "which helps K-Means calculate distances more effectively."
)

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    df[['petal length (cm)', 'petal width (cm)']]
)

km_scaled = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

scaled_cluster = km_scaled.fit_predict(scaled_data)

fig3, ax3 = plt.subplots(figsize=(7,6))

ax3.scatter(
    scaled_data[:,0],
    scaled_data[:,1],
    c=scaled_cluster,
    cmap='rainbow',
    s=70
)

ax3.scatter(
    km_scaled.cluster_centers_[:,0],
    km_scaled.cluster_centers_[:,1],
    color='black',
    marker='*',
    s=300,
    label='Centroids'
)

ax3.set_xlabel("Scaled Petal Length")
ax3.set_ylabel("Scaled Petal Width")
ax3.set_title("K-Means after Feature Scaling")
ax3.legend()

st.pyplot(fig3)

st.divider()

# -----------------------------
# Elbow Method
# -----------------------------
st.header("📉 Elbow Method")

sse = []

K = range(1,11)

for i in K:
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_data)

    sse.append(model.inertia_)

fig4, ax4 = plt.subplots(figsize=(8,5))

ax4.plot(K, sse, marker='o', linewidth=2)

ax4.set_xlabel("Number of Clusters (K)")
ax4.set_ylabel("SSE")
ax4.set_title("Elbow Method")
ax4.grid(True)

st.pyplot(fig4)

st.success("✅ Optimal value of K is approximately **3**.")

st.divider()

# -----------------------------
# Project Workflow
# -----------------------------
st.header("⚙️ Project Workflow")

st.markdown("""
1. Load Iris Dataset
2. Select Petal Length & Petal Width
3. Visualize Dataset
4. Apply K-Means Clustering
5. Perform Feature Scaling
6. Apply K-Means Again
7. Compute SSE for K = 1 to 10
8. Plot Elbow Curve
9. Determine Optimal Number of Clusters
""")

st.divider()

# -----------------------------
# About Developer
# -----------------------------
st.header("👨‍💻 About Developer")

st.markdown("""
### **Abhay Kumar Gupta**

🎓 B.Tech Computer Science & Engineering Student

💻 Passionate about Machine Learning, Data Science, Python, and AI.

---

### 🔗 Connect with Me

🐙 **GitHub**

https://github.com/Abhay-cody

💼 **LinkedIn**

https://www.linkedin.com/in/abhay-kumar-gupta-104a18397
""")

st.divider()

st.markdown(
    """
<div style='text-align:center'>
<h4>Made with ❤️ using Streamlit</h4>
<p>© 2026 Abhay Kumar Gupta</p>
</div>
""",
    unsafe_allow_html=True
)
