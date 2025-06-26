import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.linear_model import LinearRegression



with open('model\liner_regression_model.pkl','rb') as f:
    loaded_model=pickle.load(f)

st.title('E-Commerce Customer Data ')
Email= st.text_input('Enter your Email-Id')
Address=st.text_area('Enter your address')
Avatar=st.text_input('Enter your Avatar')

Avg_Session_Length=st.number_input('Average Session Length')	
Time_on_App=st.number_input('Time on App')
Time_on_Website=st.number_input('Time on Website')	
Length_of_Membership=st.number_input('Length of Membership')

if st.button('Prediction'):
    data=np.array([Avg_Session_Length,Time_on_App,Time_on_Website,Length_of_Membership]).reshape(1,-1)
    prediction=loaded_model.predict(data)
    st.write('Yearly Amount Spent: ',prediction[0])