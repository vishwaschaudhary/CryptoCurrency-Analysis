import streamlit as st
from sklearn.model_selection import train_test_split

def linearregression(X,y):
    from sklearn.linear_model import LinearRegression
    #parameter selection form
    with st.form("Linear regression form"):
        col1,col2=st.columns(2)
        p_fit_intercept=col1.radio("`fit_intercept`",[True,False])
        p_copy_X=col2.radio("`copy_X`",[True,False])
        p_n_jobs=col1.slider("`n_jobs`",-1,10,1,1)
        p_positive=col2.radio("`positive`",[False,True])
        param_train_size=col1.slider("`train_size`",10,90,70,10)
        x=st.form_submit_button("Train model")
        if x:
            X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=param_train_size)
            lr=LinearRegression(fit_intercept=p_fit_intercept,
                                copy_X=p_copy_X,
                                n_jobs=p_n_jobs,
                                positive=p_positive)
            lr.fit(X_train,y_train)
            y_pred_train=lr.predict(X_train)
            y_pred_test=lr.predict(X_test)
            from metrics import regression_metrics

            st.subheader('2. Model Performance')
            st.markdown('**2.1. Training set**')
            regression_metrics(y_train,y_pred_train)
            st.markdown("**2.2. Test set**")
            regression_metrics(y_test, y_pred_test)

            st.subheader('3. Model Parameters')
            st.write(lr.get_params())




def svr(X,y):
    from sklearn.svm import SVR
    
    with st.form("SVR parameter form"):
        col1,col2=st.columns(2)
        p_kernel=col1.selectbox("`kernel`",['rbf','linear','poly','sigmoid','precomputed'])
        if p_kernel=="poly":
            p_degree=col1.slider("`degree`",1,10,3,1)
        p_tol=col2.slider("`tol`",0.0,2.0,0.0,0.1)
        p_C=col1.slider("`C`",0.0,10.0,1.0,0.2)
        p_epsilon=col2.slider("`epsilon`",0.0,3,0.1,0.1)
        param_train_size=col1.slider("`train_size`",10,90,70,10)      
        x=st.form_submit_button("Train model")
    if x:
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=param_train_size)
        svr=SVR(kernel=p_kernel,
                degree=p_degree,
                tol=p_tol,
                C=p_C,
                epsilon=p_epsilon)
        svr.fit(X_train,y_train)
        y_pred_train=svr.predict(X_train)
        y_pred_test=svr.predict(X_test)
        from metrics import regression_metrics

        st.subheader('2. Model Performance')
        st.markdown('**2.1. Training set**')
        regression_metrics(y_train,y_pred_train)
        st.markdown("**2.2. Test set**")
        regression_metrics(y_test, y_pred_test)

        st.subheader('3. Model Parameters')
        st.write(svr.get_params())




        

def decisiontreeregressor(X,y):
    from sklearn.tree import DecisionTreeRegressor
    pass



def randomforestregressor(X,y):
    from sklearn.ensemble import RandomForestRegressor
    

    #parameter selection form:
    with st.form("Tune your Parameters"):
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
        parameter_bootstrap=col1.selectbox("`bootsrap`",[False,True])
        if parameter_bootstrap:
            parameter_oob_score=col1.selectbox("`oob_score`",[True,False])
        else:
            parameter_oob_score=False
        param_train_size=col2.slider("`train_size`",10,90,70,10)
        p_n_jobs=col2.slider("`n_jobs`",-1,10,1,1)
        x=st.form_submit_button("Train model")

    if x:
        X_train,X_test, y_train,y_test=train_test_split(X,y,train_size=param_train_size)
        rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
            max_depth=parameter_max_depth,
            max_features=parameter_max_features,
            criterion=parameter_criterion,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            bootstrap=parameter_bootstrap,
            oob_score=parameter_oob_score,
            n_jobs=p_n_jobs)
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
    

    

