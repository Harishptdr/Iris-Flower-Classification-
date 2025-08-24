import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

st.title("Iris Flower Classification")

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

sepal_length = st.slider("Enter the Sepal Length (cm):", 4.0, 8.0, 5.0)
sepal_width = st.slider("Enter the Sepal Width (cm):", 2.0, 5.0, 3.0)
petal_length = st.slider("Enter the Petal Length (cm):", 1.0, 7.0, 4.0) 
petal_width = st.slider("Enter the Petal Width (cm):", 0.1, 2.5, 1.0)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_data = scaler.transform(input_data)
predicted_class = model.predict(input_data)
st.subheader("Predicted Class: " + iris.target_names[predicted_class][0])

evaluate = st.checkbox("Evaluate Model")
if evaluate:
    st.subheader("Model Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))
    st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

visualize = st.checkbox("Visualize Data")
if visualize:
    iris = sns.load_dataset("iris")
    fig = sns.pairplot(iris, hue="species")
    st.pyplot(fig)
