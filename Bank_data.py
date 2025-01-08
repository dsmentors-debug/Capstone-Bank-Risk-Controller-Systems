
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title='Bank Risk Controller Systems', page_icon=':bank:', layout='wide')

with open("C:/Users/SG_LENOVO/Downloads/Finalxg.pkl", "rb") as model:
    xgmodel1 = pickle.load(model)

with open("C:/Users/SG_LENOVO/Downloads/label_encoder.pkl", "rb") as lemodel:
    le = pickle.load(lemodel)
    print(type(le))



df = pd.read_csv("C:/Users/SG_LENOVO/Downloads/data.csv", index_col=0)
eda_data = pd.read_csv("C:/Users/SG_LENOVO/Downloads/cleaned_data01.csv")

if 'active_section' not in st.session_state:
    st.session_state.active_section = "Home"  # Default to "Home"


st.markdown(
    """
    <style>
    /* Global page background */
    body {
        background-color: #f5f7fa;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50; /* Dark sidebar */
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ecf0f1; /* Sidebar headers */
        font-size: 26px;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1abc9c;  /* Green buttons */
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%; /* Full-width buttons */
        height: 45px; /* Equal height for all buttons */
        font-size: 26px;
        margin-bottom: 10px; /* Add spacing between buttons */
        transition: all 0.3s ease-in-out;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #16a085; /* Slightly darker green on hover */
    }
    [data-testid="stSidebar"] .stButton > button:active {
        background-color:rgb(0, 0, 0); /* Red when active/clicked */
    }

    /* Main page buttons */
    .stButton > button {
        background-color: #3498db; /* Blue buttons */
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        margin: 5px 0; /* Add some spacing */
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #2980b9; /* Darker blue on hover */
    }
    .stButton > button:active {
        background-color:rgb(0, 0, 0); /* Red when active/clicked */
    }

    /* DataFrame and table styling */
    .stDataFrame, .stTable {
        border: 1px solid #fabbbb;
        border-radius: 8px;
        background-color: white;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 14px 16px rgba(0, 0, 0, 0.1);
    }

    /* Styling for Markdown text */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
        font-family: Arial, sans-serif;
    }
    .stMarkdown p {
        color: #34495e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Bank Loan Prediction")
    st.header("Navigation")
    if st.button("Home"):
        st.session_state.active_section = "Home"
    if st.button("Data"):
        st.session_state.active_section = "Data"
    if st.button("EDA Visual"):
        st.session_state.active_section = "EDA Visual"
    if st.button("Prediction"):
        st.session_state.active_section = "Prediction"

if st.session_state.active_section == "Home":
    st.title("Welcome to the Bank Loan Prediction App")
    st.markdown("""
        This application uses machine learning to predict whether a customer will be approved for a bank loan based on certain features.
        Use the sidebar to navigate between different sections of the app.
    """)

if st.session_state.active_section == "Data":
    st.title("Dataset")
    st.write("Here is the dataset used for loan prediction:")
    st.dataframe(eda_data.head(20))  # Display first 5 rows of the dataframe

    if st.checkbox("Show Raw Data"):
        st.write(df)

if st.session_state.active_section == "EDA Visual":
    st.title("Exploratory Data Analysis (EDA) Visualizations")
    
    st.subheader("Distribution of Loan Status")
    fig = px.histogram(eda_data, x='TARGET', title="Target Distribution")
    st.plotly_chart(fig)

    Target_count = eda_data["TARGET"].value_counts().reset_index()
    Target_count.columns = ["TARGET", "count"]
    colors = ["#ffe5ff", "#ADFF2F"]

    # Process data for visualization
    gender_count = eda_data["CODE_GENDER"].value_counts().reset_index()
    gender_count.columns = ["CODE_GENDER", "count"]
    colors = ["#362A47", "#00c89b"]
    
    # Create the Plotly figure for CODE_GENDER
    fig = px.bar(
        gender_count,
        x="CODE_GENDER",
        y="count",
        title="Gender Count",
        color="CODE_GENDER",
        color_discrete_sequence=colors
    )


    st.plotly_chart(fig)
    
    # Define the function to plot distributions for categorical columns
    def plot(col, colors):
        # Generate the value counts for the column
        df = df[col].value_counts().reset_index()
        df.columns = [col, "count"]
        
        # Create a Plotly bar chart
        fig = px.bar(
            df, 
            x=col,
            y="count",
            title=f"Distribution of {col}",
            color=col,
            color_discrete_sequence=colors
        )
        
        # Display the figure in Streamlit
        st.plotly_chart(fig)
    
    # Get all categorical columns
    cat_col = df.select_dtypes(include="object").columns
    
    # Iterate over categorical columns and plot, excluding "CODE_GENDER"
    for col in cat_col:
        if col != "CODE_GENDER":  # Skip "CODE_GENDER" to avoid duplicate plot
            plot(col, colors)
    total_sum=df[["CODE_GENDER","AMT_INCOME_TOTAL"]].groupby("CODE_GENDER").agg(
    Total_income =("AMT_INCOME_TOTAL","sum"),
    Gender_count=("CODE_GENDER","count")
     ).reset_index()
    fig = px.bar(
        total_sum,
        x="CODE_GENDER",
        y="Total_income",
        title="Total Income and Gender Count by Gender",
        labels={"Total_income": "Total Income"},
        text="Total_income",  # Display the exact income values on bars
        color="CODE_GENDER",
        color_discrete_sequence=["#ffe5ff", "#00c89b"]
    )
    
    
    # Display the chart in Streamlit
    st.plotly_chart(fig)	

    

if st.session_state.active_section == "Prediction":
    st.title("Bank Loan Prediction")

    # Collect user input for prediction
    st.subheader("Enter Applicant Information")
    
    # Assuming the model requires these features
    AMT_ANNUITY_x = st.number_input("AMT_ANNUITY_x:")
    DAYS_EMPLOYED = st.number_input("DAYS_EMPLOYED:", min_value=100, max_value=2000, value=200)
    AMT_CREDIT_x = st.number_input("AMT_CREDIT_x:")
    AMT_INCOME_TOTAL = st.number_input("AMT_INCOME_TOTAL:")
    ORGANIZATION_TYPE = st.number_input("ORGANIZATION_TYPE:", min_value=1, max_value=50, value=2)
    AMT_REQ_CREDIT_BUREAU_YEAR = st.number_input("AMT_REQ_CREDIT_BUREAU_YEAR:", min_value=0, max_value=23, value=1)
    DAYS_LAST_DUE = st.number_input("DAYS_LAST_DUE:", min_value=25, max_value=2000, value=100)
    OCCUPATION_TYPE = st.number_input("OCCUPATION_TYPE:", min_value=1, max_value=18, value=2)
    AMT_REQ_CREDIT_BUREAU_MON = st.number_input("AMT_REQ_CREDIT_BUREAU_MON:", min_value=0, max_value=23, value=1)
    FLAG_OWN_REALTY = st.selectbox("FLAG_OWN_REALTY:", [1, 0])
    NAME_CASH_LOAN_PURPOSE = st.number_input("NAME_CASH_LOAN_PURPOSE:", min_value=0, max_value=24, value=23)
    NAME_CLIENT_TYPE = st.selectbox("NAME_CLIENT_TYPE:", [0, 1, 2, 3, 4, 5])
    Age_group = st.selectbox("Age_group:", [1, 0, 2])
    AMT_REQ_CREDIT_BUREAU_WEEK = st.number_input("AMT_REQ_CREDIT_BUREAU_WEEK:", min_value=0, max_value=23, value=1)
    NAME_INCOME_TYPE = st.selectbox("NAME_INCOME_TYPE:", [6, 0, 2, 3, 5, 4, 1])

    # Encoding categorical features


    ORGANIZATION_TYPE = le.get(ORGANIZATION_TYPE,-1)
    OCCUPATION_TYPE = le.get(OCCUPATION_TYPE,-1)
    FLAG_OWN_REALTY = le.get(FLAG_OWN_REALTY,-1)
    NAME_CLIENT_TYPE=le.get(NAME_CLIENT_TYPE,-1)
    Age_group=le.get(Age_group,-1)
    NAME_INCOME_TYPE=le.get(NAME_INCOME_TYPE,-1)

    # Prepare input data
    input_data = pd.DataFrame({
        'AMT_ANNUITY_x': [AMT_ANNUITY_x],
        'DAYS_EMPLOYED': [DAYS_EMPLOYED],
        'AMT_CREDIT_x': [AMT_CREDIT_x],
        'AMT_INCOME_TOTAL': [AMT_INCOME_TOTAL],
        'ORGANIZATION_TYPE': [ORGANIZATION_TYPE],
        'AMT_REQ_CREDIT_BUREAU_YEAR': [AMT_REQ_CREDIT_BUREAU_YEAR],
        'DAYS_LAST_DUE': [DAYS_LAST_DUE],
        'OCCUPATION_TYPE': [OCCUPATION_TYPE],
        'AMT_REQ_CREDIT_BUREAU_MON':[AMT_REQ_CREDIT_BUREAU_MON],
        'FLAG_OWN_REALTY':[FLAG_OWN_REALTY],
        'NAME_CASH_LOAN_PURPOSE':[NAME_CASH_LOAN_PURPOSE],
        'NAME_CLIENT_TYPE':[NAME_CLIENT_TYPE],
        'Age_group':[Age_group],
        'AMT_REQ_CREDIT_BUREAU_WEEK':[AMT_REQ_CREDIT_BUREAU_WEEK],
        'NAME_INCOME_TYPE':[NAME_INCOME_TYPE]
    })

    # Standardize the input data using the same scaler used for training
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Prediction
    if st.button("Predict"):
        prediction = xgmodel1.predict(input_scaled)
        prediction = 'Approved' if prediction[0] == 1 else 'Rejected'
        st.write(f"Prediction: The loan application is {prediction}")




