import streamlit as st
import pandas as pd


st.set_page_config(page_title="TweakML",
                   layout="wide")


st.write("""
# TweakML

The perfect place to easily train different Machine Learning models according to your required task, tweak their hyperparameters and shortlist the best models.


""")

task=st.radio("Select any one option:",options=["Regression","Binary Classification","Multi-class Classification"])

st.markdown('#### Upload your CSV data')
uploaded_file = st.file_uploader("Please upload preprocessed data with last column as target variable", type=["csv"])

def after_data_input(df):
    if task=="Regression":
        st.write("Choose your model to train:")
        algorithm=st.radio("",options=["LinearRegression","SVR","DecisionTreeRegressor","RandomForestRegressor"])
        
        if algorithm=="LinearRegression":
            from models.Regression import linearregression
            linearregression(df)
        elif algorithm=="SVR":
            from models.Regression import svr
            svr(df)
        elif algorithm=="DecisionTreeRegressor":
            from models.Regression import decisiontreeregressor
            decisiontreeregressor(df)
        else:
            from models.Regression import randomforestregressor
            randomforestregressor(df)

    elif task=="Binary Classification":
        st.write("Choose your model to train:")
        algorithm=st.radio("",options=["LogisticRegression","SVC","DecisionTreeClassifier","RandomForestClassifier"])
        
        if algorithm=="LogisticRegression":
            from models.Regression import logisticregression
            logisticregression(df)
        elif algorithm=="SVC":
            from models.Regression import svc
            svc(df)
        elif algorithm=="DecisionTreeClassifier":
            from models.Regression import decisiontreeclassifier
            decisiontreeclassifier(df)
        else:
            from models.Regression import randomforestclassifier
            randomforestclassifier(df)

    

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head(5))
    after_data_input(df)
    

else:
    if st.button('Press to use Example Dataset'):
        df = pd.read_csv("example_data/Real estate.csv")
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df.head(5))
        after_data_input(df)
        


