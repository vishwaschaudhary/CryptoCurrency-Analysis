import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston

st.set_page_config(page_title="HyperTuner",
                   layout="wide")

def build_model(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]

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

    X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=(100-split_size)/100)
    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=-1)
    rf.fit(X_train, y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    y_pred_train=rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_train, y_pred_train))

    st.write("Error")
    st.info(mean_squared_error(y_train,y_pred_train))
    
    st.markdown("**2.2. Test set**")
    y_pred_test=rf.predict(X_test)
    st.write("Coefficient of determination ($R^2$):")
    st.info(r2_score(y_test,y_pred_test))
    st.write("Error")
    st.info(mean_squared_error(y_test,y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())


st.write("""
# The Machine Learning App

In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.

Try adjusting the hyperparameters!

""")

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")
    
# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    
#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df) 

