import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# ----------------------------
# Load Model & Files
# ----------------------------
model = tf.keras.models.load_model("churn_model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Customer Churn Prediction")

# Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 30)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", value=0.0)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict"):

    # Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Convert Yes/No to 1/0
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0

    # Geography One Hot Encoding
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_columns = onehot_encoder_geo.get_feature_names_out(["Geography"])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

    # Main DataFrame
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # Merge Geography columns
    final_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Arrange same columns as scaler trained on
    final_df = final_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale Input
    input_scaled = scaler.transform(final_df)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = prediction[0][0]

    st.write(f"Churn Probability: {probability:.2f}")

    if probability > 0.5:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")