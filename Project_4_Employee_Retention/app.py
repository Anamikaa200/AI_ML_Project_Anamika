import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
.main{
    padding-top:20px;
}

.title{
    text-align:center;
    color:#1f77b4;
}

.footer{
    text-align:center;
    padding:20px;
    border-top:1px solid #ddd;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.markdown("<h1 class='title'>Employee Retention Prediction using Logistic Regression</h1>",
unsafe_allow_html=True)

st.write(
"""
This project predicts whether an employee is likely to leave the company
using a **Logistic Regression Machine Learning Model**.
"""
)

# ---------------------------
# Load Dataset
# ---------------------------
from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    from pathlib import Path
    import pandas as pd

    BASE_DIR = Path(__file__).parent
    csv_path = BASE_DIR / "HR_comma_sep.csv"

    return pd.read_csv(csv_path)

# Load dataset
df = load_data()

# ---------------------------
# Sidebar
# ---------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset",
        "EDA",
        "Visualizations",
        "Model",
        "Prediction"
    ]
)

# ==========================
# DATASET
# ==========================
if menu=="Dataset":

    st.header("Dataset Preview")

    st.dataframe(df)

    st.subheader("Shape")

    st.write(df.shape)

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    st.write(df.isnull().sum())


# ==========================
# EDA
# ==========================
elif menu=="EDA":

    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))

    corr=df.corr(numeric_only=True)

    im=ax.imshow(corr)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    plt.colorbar(im)

    st.pyplot(fig)


# ==========================
# VISUALIZATION
# ==========================
elif menu=="Visualizations":

    st.header("Employee Retention Visualizations")

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Salary vs Retention")

        salary=pd.crosstab(df.salary,df.left)

        fig,ax=plt.subplots()

        salary.plot(kind="bar",ax=ax)

        st.pyplot(fig)

    with col2:

        st.subheader("Department vs Retention")

        dept=pd.crosstab(df.Department,df.left)

        fig,ax=plt.subplots(figsize=(8,5))

        dept.plot(kind="bar",ax=ax)

        st.pyplot(fig)


# ==========================
# MODEL
# ==========================
elif menu=="Model":

    st.header("Logistic Regression Model")

    X=df[['satisfaction_level',
          'average_montly_hours',
          'promotion_last_5years',
          'salary']]

    salary_dummy=pd.get_dummies(
        X['salary'],
        prefix="salary",
        drop_first=True
    )

    X=pd.concat([

        X.drop("salary",axis=1),

        salary_dummy

    ],axis=1)

    y=df.left

    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model=LogisticRegression(max_iter=1000)

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    acc=accuracy_score(y_test,pred)

    st.success(f"Model Accuracy : {acc*100:.2f}%")

    st.subheader("Confusion Matrix")

    cm=confusion_matrix(y_test,pred)

    fig,ax=plt.subplots()

    im=ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.colorbar(im)

    st.pyplot(fig)

    st.subheader("Classification Report")

    st.text(classification_report(y_test,pred))


# ==========================
# PREDICTION
# ==========================
else:

    st.header("Employee Retention Prediction")

    satisfaction=st.slider(
        "Satisfaction Level",
        0.0,
        1.0,
        0.5
    )

    hours=st.slider(
        "Average Monthly Hours",
        90,
        320,
        200
    )

    promotion=st.selectbox(
        "Promotion in Last 5 Years",
        [0,1]
    )

    salary=st.selectbox(
        "Salary",
        ["low","medium","high"]
    )

    X=df[['satisfaction_level',
          'average_montly_hours',
          'promotion_last_5years',
          'salary']]

    salary_dummy=pd.get_dummies(
        X.salary,
        prefix="salary",
        drop_first=True
    )

    X=pd.concat(
        [
            X.drop("salary",axis=1),
            salary_dummy
        ],
        axis=1
    )

    y=df.left

    model=LogisticRegression(max_iter=1000)

    model.fit(X,y)

    sample=pd.DataFrame({
        "satisfaction_level":[satisfaction],
        "average_montly_hours":[hours],
        "promotion_last_5years":[promotion],
        "salary_low":[1 if salary=="low" else 0],
        "salary_medium":[1 if salary=="medium" else 0]
    })

    prediction=model.predict(sample)[0]

    if prediction==1:
        st.error("⚠ Employee is likely to leave the company.")
    else:
        st.success("✅ Employee is likely to stay in the company.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")

st.markdown("""
<div class="footer">

### 👨‍💻 Developer

**Abhay Kumar Gupta**

🔗 <a href="https://github.com/Abhay-cody" target="_blank">GitHub</a>

🔗 <a href="https://www.linkedin.com/in/abhay-kumar-gupta-104a18397" target="_blank">LinkedIn</a>

</div>
""", unsafe_allow_html=True)
