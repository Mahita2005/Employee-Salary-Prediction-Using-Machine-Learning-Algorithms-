import streamlit as st
import numpy as np
import joblib


model = joblib.load("best_salary_model.pkl")
scaler = joblib.load("scaler.pkl")

encoders = {
    'workclass': joblib.load("workclass_.pkl"),
    'marital-status': joblib.load("marital-status_.pkl"),
    'occupation': joblib.load("occupation_.pkl"),
    'relationship': joblib.load("relationship_.pkl"),
    'race': joblib.load("race_.pkl"),
    'gender': joblib.load("gender_.pkl"),
    'native-country': joblib.load("native-country_.pkl")
}


st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .css-18e3th9 {
        background-color: #0e1117;
        color: white;
    }
    footer {
        visibility: hidden;
    }
    .footer:after {
        content: 'Made by Mahita';
        visibility: visible;
        display: block;
        position: relative;
        color: gray;
        text-align: center;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Employee Salary Prediction App")
st.write("Enter employee details below to predict if salary is >50K or <=50K")


with st.form("prediction_form"):
    age = st.slider("Age", 18, 70, 30)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 50000)
    education_num = st.slider("Education Number", 1, 16, 10)
    
    workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
    marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
    occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
    relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
    race = st.selectbox("Race", encoders['race'].classes_)
    gender = st.selectbox("Gender", encoders['gender'].classes_)
    native_country = st.selectbox("Native Country", encoders['native-country'].classes_)
    
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0)

    submit = st.form_submit_button("Predict")

if submit:
    try:
        input_data = [
            age,
            fnlwgt,
            education_num,
            encoders['workclass'].transform([workclass])[0],
            encoders['marital-status'].transform([marital_status])[0],
            encoders['occupation'].transform([occupation])[0],
            encoders['relationship'].transform([relationship])[0],
            encoders['race'].transform([race])[0],
            encoders['gender'].transform([gender])[0],
            hours_per_week,
            encoders['native-country'].transform([native_country])[0],
            capital_gain,
            capital_loss
        ]

        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]

        st.subheader("Prediction Result:")
        if prediction == 1 or prediction == '>50K':
            st.success("The predicted salary is **greater than 50K**.")
        else:
            st.warning("The predicted salary is **less than or equal to 50K**.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")


st.markdown(
    """
    <div class="footer"></div>
    """,
    unsafe_allow_html=True
)
