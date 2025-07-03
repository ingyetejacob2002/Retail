import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load(r'C:\Users\user\Desktop\item prediction\pipeline.joblib')

# App title and header
st.title("🛍️ Retail Store Return Prediction")
st.markdown("Use this app to predict **Item Store Returns** based on product and store characteristics.")

# Form for input
with st.form("prediction_form"):
    st.subheader("📋 Enter Item and Store Details")

    item_weight = st.number_input("Item Weight", min_value=0.0, max_value=50.0, step=0.1)
    item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, step=0.01)
    item_price = st.number_input("Item Price", min_value=0.0, max_value=500.0, step=1.0)
    store_year = st.number_input("Store Start Year", min_value=1950, max_value=2030, step=1)

    item_sugar_content = st.selectbox("Item Sugar Content", ['Low', 'Medium', 'High'])
    item_type = st.selectbox("Item Type", ['Food', 'Drinks', 'Household'])  # match training values
    store_size = st.selectbox("Store Size", ['Small', 'Medium', 'High'])
    store_location = st.selectbox("Store Location Type", ['Urban', 'Suburban', 'Rural'])
    store_type = st.selectbox("Store Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Create input DataFrame
        input_df = pd.DataFrame({
            'Item_Weight': [item_weight],
            'Item_Visibility': [item_visibility],
            'Item_Price': [item_price],
            'Store_Start_Year': [store_year],
            'Item_Sugar_Content': [item_sugar_content],
            'Item_Type': [item_type],
            'Store_Size': [store_size],
            'Store_Location_Type': [store_location],
            'Store_Type': [store_type],
        })

        # Make prediction
        try:
            prediction = pipeline.predict(input_df)[0]
            st.success(f"📊 Predicted Item Store Return: **{prediction:.2f}**")
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
