import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

#Load the trained model
model = load_model('model.h5')

with open('label_encoder.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder.pkl','rb') as f:
    one_hot_geo = pickle.load(f)

with open('standard_scaler.pkl','rb') as f:
    scaler = pickle.load(f)

#Streamlit app
st.title('Churn Prediction App')

#Input fields
geography = st.selectbox('Geography',one_hot_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#One-hot encode the geography
geo_encoded = one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=['France', 'Germany', 'Spain'])

input_df = pd.concat([input_data.drop('Geography',axis=1), geo_encoded_df], axis=1)

#Scale the input data
input_scaled = scaler.transform(input_df)

#Make the prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

#Display the result
if prediction_proba > 0.5:
    st.write(f'Prediction: The customer is likely to churn with a probability of {prediction_proba:.2f}')
else:
    st.write(f'Prediction: The customer is not likely to churn with a probability of {prediction_proba:.2f}')


