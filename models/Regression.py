import streamlit as st
from sklearn.model_selection import train_test_split

def linearregression(df):
    pass

def svr(df):
    pass

def decisiontreeregressor(df):
    pass



def randomforestregressor(df):
    from sklearn.ensemble import RandomForestRegressor
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

    #parameter selection form:
    with st.form("Parameter form"):
        col1,col2=st.columns(2)
        parameter_n_estimators=col1.slider("`n_estimators`",10,500,100,10)
        parameter_max_depth=col2.slider("`max_depth`",1,150,5,2)
        parameter_max_features_select=col1.selectbox("`max_features`",options=["int","float","sqrt","log2"])
        if parameter_max_features_select=="int":
            parameter_max_features=col1.slider("",1,10,1,1)
        elif parameter_max_features_select=="float":
            parameter_max_features=col1.slider("",0.1,1,0.1,0.1)
        else:
            parameter_max_features=parameter_max_features_select
        parameter_criterion=col2.selectbox("`criterion`",["squared_error", "absolute_error", "friedman_mse", "poisson"])
        parameter_min_samples_split=col1.slider("`min_samples_split`",1,100,2,1)
        parameter_min_samples_leaf=col2.slider("`min_samples_leaf`",1,10,1,1)
        parameter_bootstrap=col1.selectbox("`bootsrap`",[True,False])
        if parameter_bootstrap:
            parameter_oob_score=col1.selectbox("`oob_score`",[True,False])
        else:
            parameter_oob_score=False
        param_train_size=col2.slider("`train_size`",10,90,10)


    X_train,X_test, y_train,y_test=train_test_split(X,y,train_size=param_train_size)
    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        max_depth=parameter_max_depth,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred_train=rf.predict(X_train)
    y_pred_test=rf.predict(X_test)
    from metrics import regression_metrics

    st.subheader('2. Model Performance')
    st.markdown('**2.1. Training set**')
    regression_metrics(y_train,y_pred_train)
    st.markdown("**2.2. Test set**")
    regression_metrics(y_test, y_pred_test)

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())
    

    

