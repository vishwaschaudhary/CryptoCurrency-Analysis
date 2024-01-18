import streamlit as st
from sklearn.model_selection import train_test_split

def logisticregression(X,y):
    from sklearn.linear_model import LogisticRegression
    with st.form("logistic regression parameters"):
        col1,col2=st.columns(2)
        p_penalty=col1.selectbox("`penalty`",["l2","l1","elasticnet",None])
        p_C=col2.slider("`C`",0,10,1,1)
        p_solver=col1.selectbox("`solver`",["lbfgs","liblinear","newton-cg","newton-cholesky","sag","saga"])
        p_max_iter=col2.slider("`max_iter`",50,200,100,5)
        p_multiclass=col1.selectbox("`multi_class`",["auto","ovr","multinomial"])
        p_warm_start=col2.select_slider("`warm_start`",[False,True])
        param_train_size=col1.slider("`train_size`",10,90,70,10)
        x=st.form_submit_button("Train model")

    if x:
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=param_train_size)
        lg=LogisticRegression(penalty=p_penalty,
                              C=p_C,
                              solver=p_solver,
                              max_iter=p_max_iter,
                              multi_class=p_multiclass,
                              warm_start=p_warm_start)
        lg.fit(X_train,y_train)
        y_pred_train=lg.predict(X_train)
        y_pred_test=lg.predict(X_test)
        from metrics import classification_metrics

        st.subheader('2. Model Performance')
        st.markdown('**2.1. Training set**')
        classification_metrics(y_train,y_pred_train)
        st.markdown("**2.2. Test set**")
        classification_metrics(y_test, y_pred_test)
        st.subheader('3. Model Parameters')
        st.write(lg.get_params())
    



def svc(X,y):
    from sklearn.svm import SVC
    with st.form("SVC parameter form"):
        col1,col2=st.columns(2)
        p_kernel=col1.selectbox("`kernel`",['rbf','linear','poly','sigmoid','precomputed'])
        p_tol=col2.slider("`tol`",0.1,2.0,0.1,0.1)
        p_C=col1.slider("`C`",0.0,10.0,1.0,0.2)
        p_epsilon=col2.slider("`epsilon`",0.0,3.0,0.1,0.1)
        param_train_size=col1.slider("`train_size`",10,90,70,10)      
        x=st.form_submit_button("Train model")
    if x:
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=param_train_size)
        svc=SVC(kernel=p_kernel,
                tol=p_tol,
                C=p_C)

        svc.fit(X_train,y_train)
        y_pred_train=svc.predict(X_train)
        y_pred_test=svc.predict(X_test)
        from metrics import classification_metrics

        st.subheader('2. Model Performance')
        st.markdown('**2.1. Training set**')
        classification_metrics(y_train,y_pred_train)
        st.markdown("**2.2. Test set**")
        classification_metrics(y_test, y_pred_test)
        st.subheader('3. Model Parameters')
        st.write(svc.get_params())

def decisiontreeclassifier(X,y):
    from sklearn.tree import DecisionTreeClassifier
    with st.form("decisionnn tree parameters"):
        col1,col2=st.columns(2)
        parameter_max_depth=col2.slider("`max_depth`",1,50,5,2)
        parameter_criterion=col2.selectbox("`criterion`",["gini", "entropy", "log_loss"])
        parameter_min_samples_split=col1.slider("`min_samples_split`",1,100,2,1)
        parameter_min_samples_leaf=col2.slider("`min_samples_leaf`",1,10,1,1)
        param_train_size=col1.slider("`train_size`",10,90,70,10)
        p_splitter=col1.select_slider("`splitter`",["best","random"])
        x=st.form_submit_button("Train model")

    if x:
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=param_train_size)
        dtc=DecisionTreeClassifier(criterion=parameter_criterion,
                                  max_depth=parameter_max_depth,
                                  min_samples_split=parameter_min_samples_split,
                                  min_samples_leaf=parameter_min_samples_leaf,
                                  splitter=p_splitter)
        dtc.fit(X_train,y_train)
        y_pred_train=dtc.predict(X_train)
        y_pred_test=dtc.predict(X_test)
        from metrics import classification_metrics

        st.subheader('2. Model Performance')
        st.markdown('**2.1. Training set**')
        classification_metrics(y_train,y_pred_train)
        st.markdown("**2.2. Test set**")
        classification_metrics(y_test, y_pred_test)

        st.subheader('3. Model Parameters')
        st.write(dtc.get_params())

def randomforestclassifier(X,y):
    from sklearn.ensemble import RandomForestClassifier
    

    #parameter selection form:
    with st.form("Tune your random forest Parameters"):
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
        parameter_criterion=col2.selectbox("`criterion`",["gini", "entropy", "log_loss"])
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
        rf = RandomForestClassifier(n_estimators=parameter_n_estimators,
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
        from metrics import classification_metrics

        st.subheader('2. Model Performance')
        st.markdown('**2.1. Training set**')
        classification_metrics(y_train,y_pred_train)
        st.markdown("**2.2. Test set**")
        classification_metrics(y_test, y_pred_test)

        st.subheader('3. Model Parameters')
        st.write(rf.get_params())
    