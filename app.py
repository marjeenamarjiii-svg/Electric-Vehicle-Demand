import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load model and scaler
# -----------------------------
model = pickle.load(open("ev_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# App title
# -----------------------------
st.title("🚗 Electric Vehicle Demand Prediction")

st.write("Enter values to predict EV demand")

# -----------------------------
# Load dataset to get column names
# -----------------------------
df = pd.read_csv("ev.csv")

df['Target'] = np.where(
    df['EV_Sales_Quantity'] > df['EV_Sales_Quantity'].median(),
    1,
    0
)

X = df.drop(['Target'], axis=1)
X = pd.get_dummies(X)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Features")

input_data = {}

for col in X.columns:
    input_data[col] = st.sidebar.number_input(f"{col}", value=0)

input_df = pd.DataFrame([input_data])

# -----------------------------
# Scaling input
# -----------------------------
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict EV Demand"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("High EV Demand Expected 🚀")
    else:
        st.warning("Low EV Demand Expected ⚠️")
