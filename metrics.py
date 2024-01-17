import streamlit as st
def classification_metrics(y_actual,y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc=accuracy_score(y_actual, y_pred)
    precision=precision_score(y_actual, y_pred)
    recall=recall_score(y_actual, y_pred)
    f1=f1_score(y_actual,y_pred)
    auc=roc_auc_score(y_actual,y_pred)
    return acc, precision, recall, f1, auc

def regression_metrics(y_actual,y_pred):
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    mse=mean_squared_error(y_actual,y_pred)
    mae=mean_absolute_error(y_actual,y_pred)
    r2=r2_score(y_actual,y_pred)
    st.write("`mean_squared_error`")
    st.info(mse)
    st.write("`mean_absolute_error`")
    st.info(mae)
    st.write("`($R^2$)`")
    st.info(r2)
