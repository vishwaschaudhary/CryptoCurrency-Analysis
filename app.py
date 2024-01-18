import streamlit as st
import pandas as pd


st.set_page_config(page_title="TweakML",
                   layout="wide")

#setting title 
st.write("""
# TweakML

The perfect place to easily train different Machine Learning models according to your required task, tweak their hyperparameters and shortlist the best models.


""")

#asking the user about the kind of task he wants 
task=st.radio("Select any one option:",options=["Regression","Classification"],key="task",index=0)
#giving a dummy dataset to uploaded_file so that the interpreter executes the code
 
#taking data from the user or asking him to use sample data

# uploaded_file 
# if uploaded_file is None:


#storing this uploaded data on once


st.write("#### Would you like to upload your dataset or proceed with example dataset:")
user_choice=st.radio("choice",options=["Upload dataset","Example dataset"])
if user_choice=="Example dataset":
    if task=="Regression":
        uploaded_file="example_data/Real estate.csv"
    elif task=="Classification":
        uploaded_file="example_data/binary classification.csv"
else: 
    uploaded_file= st.file_uploader("Please upload preprocessed data", type=["csv"])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head(5))
    target=st.selectbox("Select the target variable",options=list(df.columns))
    X=df.drop(target,axis=1)
    y=df[target]

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X.shape)
    st.write("Test set")
    st.info(y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write("X variable")
    st.info(list(X.columns))
    st.write("y variable")
    st.info(y.name)

    if task=="Regression":
        st.write("Choose your model to train:")
        

        algorithm=st.selectbox("label",options=("LinearRegression","SVR","DecisionTreeRegressor","RandomForestRegressor"),label_visibility="hidden",key="algorithm")
        
        
        if algorithm=="LinearRegression":
            from models.Regression import linearregression
            linearregression(X,y)
        elif algorithm=="SVR":
            from models.Regression import svr
            svr(X,y)
        elif algorithm=="DecisionTreeRegressor":
            from models.Regression import decisiontreeregressor
            decisiontreeregressor(X,y)
        elif algorithm=="RandomForestRegressor":
            from models.Regression import randomforestregressor
            randomforestregressor(X,y)

    elif task=="Classification":
        st.write("Choose your model to train:")
        algorithm=st.radio("label",options=["LogisticRegression","SVC","DecisionTreeClassifier","RandomForestClassifier"],label_visibility="hidden")
        
        if algorithm=="LogisticRegression":
            from models.classification import logisticregression
            logisticregression(X,y)
        elif algorithm=="SVC":
            from models.classification import svc
            svc(X,y)
        elif algorithm=="DecisionTreeClassifier":
            from models.classification import decisiontreeclassifier
            decisiontreeclassifier(X,y)
        else:
            from models.classification import randomforestclassifier
            randomforestclassifier(X,y)








    
    
    


        


