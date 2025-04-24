import os
import urllib.request
import streamlit as st
import pandas as pd
import joblib
import gdown

MODEL_URL = 'https://drive.google.com/uc?export=download&id=182udCUHZRIfGdG7YMoIJ8OROdocFVPFC'
MODEL_PATH = 'random_forest_model.pkl'

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model and encoders
model = joblib.load(MODEL_PATH)
encoders = joblib.load('label_encoders.pkl')

st.title("Home Credit Scorecard Model's App")
st.write("Author: Johanes Paulus Bernard Purek")
st.write("Fill in the client's information below to predict loan repayment risk.")

# --- Numeric Inputs ---
days_employed = st.number_input("Days Employed")
amt_goods_price = st.number_input("Goods Price")
amt_credit = st.number_input("Credit Amount")
days_birth = st.number_input("Days Since Birth")
amt_income_total = st.number_input("Annual Income")
days_registration = st.number_input("Days Since Registration")
days_last_phone_change = st.number_input("Days Since Last Phone Change")
days_id_publish = st.number_input("Days Since ID Published")
amt_annuity = st.number_input("Annuity Amount")
age = st.number_input("Age")
ext_source_2 = st.number_input("External Source 2 Score")
ext_source_3 = st.number_input("External Source 3 Score")
def_30_cnt_social_circle = st.number_input("30 Days Default Count in Social Circle")
def_60_cnt_social_circle = st.number_input("60 Days Default Count in Social Circle")

# --- Encoded Categorical Inputs ---
st.write("Advertising:(0), Agriculture:(1), Bank:(2), Business Entity Type 1:(3), Business Entity Type 2:(4), Business Entity Type 3:(5), Cleaning:(6), Construction:(7), Culture:(8), Electricity:(9), Emergency:(10), Government:(11), Hotel:(12), Housing:(13), Industry: type 1:(14), Industry: type 10:(15), Industry: type 11:(16), Industry: type 12:(17), Industry: type 13:(18), Industry: type 2:(19), Industry: type 3:(20), Industry: type 4:(21), Industry: type 5:(22), Industry: type 6:(23), Industry: type 7:(24), Industry: type 8:(25), Industry: type 9:(26), Insurance:(27), Kindergarten:(28), Legal Services:(29), Medicine:(30), Military:(31), Mobile:(32), Other:(33), Police:(34), Postal:(35), Realtor:(36), Religion:(37), Restaurant:(38), School:(39), Security:(40), Security Ministries:(41), Self-employed:(42), Services:(43), Telecom:(44), Trade: type 1:(45), Trade: type 2:(46), Trade: type 3:(47), Trade: type 4:(48), Trade: type 5:(49), Trade: type 6:(50), Trade: type 7:(51), Transport: type 1:(52), Transport: type 2:(53), Transport: type 3:(54), Transport: type 4:(55), University:(56), XNA:(57)")
organization_type = st.selectbox("Organization Type", encoders['ORGANIZATION_TYPE'].classes_)
organization_encoded = encoders['ORGANIZATION_TYPE'].transform([organization_type])[0]

st.write("Businessman:(0), Commercial associate:(1), Maternity leave:(2), Pensioner:(3), State servant:(4), Student:(5), Unemployed:(6), Working:(7)")
name_income_type = st.selectbox("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
income_encoded = encoders['NAME_INCOME_TYPE'].transform([name_income_type])[0]

st.write("Academic degree:(0), Higher education:(1), Incomplete higher:(2), Lower secondary:(3), Secondary / secondary special:(4)")
name_education_type = st.selectbox("Education Level", encoders['NAME_EDUCATION_TYPE'].classes_)
edu_encoded = encoders['NAME_EDUCATION_TYPE'].transform([name_education_type])[0]

st.write("Female:(0), Male:(1)")
code_gender = st.selectbox("Gender", encoders['CODE_GENDER'].classes_)
gender_encoded = encoders['CODE_GENDER'].transform([code_gender])[0]

# --- Binary Encoded Inputs ---
reg_city_not_work_city = st.selectbox("City of Work ≠ Registration City?", ['Yes', 'No'])
reg_city_not_work_encoded = 1 if reg_city_not_work_city == 'Yes' else 0

reg_city_not_live_city = st.selectbox("City of Living ≠ Registration City?", ['Yes', 'No'])
reg_city_not_live_encoded = 1 if reg_city_not_live_city == 'Yes' else 0

# --- Inputs ---
input_dict = {
    'DAYS_EMPLOYED': days_employed,
    'AMT_GOODS_PRICE': amt_goods_price,
    'AMT_CREDIT': amt_credit,
    'DAYS_BIRTH': days_birth,
    'AMT_INCOME_TOTAL': amt_income_total,
    'DAYS_REGISTRATION': days_registration,
    'DAYS_LAST_PHONE_CHANGE': days_last_phone_change,
    'DAYS_ID_PUBLISH': days_id_publish,
    'AMT_ANNUITY': amt_annuity,
    'AGE': age,
    'ORGANIZATION_TYPE': organization_encoded,
    'NAME_INCOME_TYPE': income_encoded,
    'REG_CITY_NOT_WORK_CITY': reg_city_not_work_encoded,
    'CODE_GENDER': gender_encoded,
    'EXT_SOURCE_2': ext_source_2,
    'REG_CITY_NOT_LIVE_CITY': reg_city_not_live_encoded,
    'NAME_EDUCATION_TYPE': edu_encoded,
    'DEF_30_CNT_SOCIAL_CIRCLE': def_30_cnt_social_circle,
    'EXT_SOURCE_3': ext_source_3,
    'DEF_60_CNT_SOCIAL_CIRCLE': def_60_cnt_social_circle
}

input_df = pd.DataFrame([input_dict])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "❌ At Risk of Repayment Difficulty" if prediction == 1 else "✅ Likely to Repay Successfully"
    st.subheader(f"Prediction Result: {result}")
    st.caption(f"Model Confidence (Risk Probability): {probability:.2%}")
