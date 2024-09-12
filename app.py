import numpy as np
import pandas as pd
import streamlit as st 
import sklearn
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import LogisticRegression

modelLR = joblib.load('model_log_reg.sav')
pipeline = joblib.load('iris_pipeline.sav')


def main():
    st.title('Clasificando flores Iris')
    st.markdown('Modelo para clasificación de flores iris en setosa, versicolor, virginica')
    st.header('Características')
    col1, col2 = st.columns(2)
    with col1:
        st.text('Características del sépalo')
        sepal_l = st.slider('Sepal lenght (cm)', 4.0, 8.0, 0.5)
        sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
    with col2:
        st.text('Características del pétalo')
        petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
        petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)
    if st.button("Predecir Flor de Iris"):
        features = [[sepal_l,sepal_w,petal_l,petal_l,petal_w]]
        data = {"SepalLengthCm":float(sepal_l),"SepalWidthCm":float(sepal_w),"PetalLengthCm":float(petal_l),"PetalWidthCm":float(petal_w)}
        df = pd.DataFrame([list(data.values())],columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])
        df_prepared = pipeline.transform(df)
        prediction = modelLR.predict(df_prepared)
        if prediction == 0:
            output = 'Iris-Setosa'
        elif prediction == 1:
            output = 'iris-Versicolor'
        elif prediction == 2:
            output = 'Iris-Virginica'
        st.success('La flor es {}'.format(output))

if __name__=='__main__': 
    main()