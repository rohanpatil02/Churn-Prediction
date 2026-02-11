import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("best_model.pkl", "rb"))
        ohe = pickle.load(open("ohe.pkl", "rb"))
        ord_internet = pickle.load(open("ord_internet.pkl", "rb"))
        ord_contract = pickle.load(open("ord_contract.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
        feature_order = pickle.load(open("feature_order.pkl", "rb"))
        return model, ohe, ord_internet, ord_contract, scaler, label_encoder, feature_order
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

model, ohe, ord_internet, ord_contract, scaler, label_encoder, feature_order = load_models()

# Title and description
st.title("📊 Telco Customer Churn Prediction")
st.markdown("### Predict whether a customer will churn based on their profile and services")
st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    st.subheader("📞 Phone Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    st.subheader("💳 Billing Information")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                  ["Electronic check", "Mailed check", 
                                   "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    st.subheader("🌐 Internet Services")
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    st.subheader("💰 Account Information")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=10.0)

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    # Create input dataframe
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input
    try:
        # Replace MultipleLines
        input_df["MultipleLines"].replace('No phone service', "No", inplace=True)
        
        # One-Hot Encoding
        ohe_cols = ["gender", "PaymentMethod", "Partner", "Dependents", "PhoneService", 
                    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
        ohe_df = pd.DataFrame(ohe.transform(input_df[ohe_cols]), columns=ohe.get_feature_names_out())
        
        # Ordinal Encoding
        input_df["InternetService"] = ord_internet.transform(input_df[["InternetService"]])
        input_df["Contract"] = ord_contract.transform(input_df[["Contract"]])
        
        # Concatenate
        input_processed = pd.concat([input_df.drop(ohe_cols, axis=1).reset_index(drop=True), 
                                     ohe_df.reset_index(drop=True)], axis=1)
        
        # Ensure correct column order
        input_processed = input_processed[feature_order]
        
        # Scale
        input_scaled = scaler.transform(input_processed)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            churn_label = label_encoder.inverse_transform([prediction])[0]
            if churn_label == "Yes":
                st.error(f"### ⚠️ Customer Will Likely CHURN")
                st.metric("Churn Probability", f"{prediction_proba[1]:.1%}")
            else:
                st.success(f"### ✅ Customer Will Likely STAY")
                st.metric("Stay Probability", f"{prediction_proba[0]:.1%}")
        
        with result_col2:
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['Stay', 'Churn'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            st.bar_chart(prob_df.set_index('Outcome'))
        
        # Detailed probabilities
        st.markdown("### Detailed Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.info(f"**Probability of Staying:** {prediction_proba[0]:.4f} ({prediction_proba[0]*100:.2f}%)")
        with prob_col2:
            st.warning(f"**Probability of Churning:** {prediction_proba[1]:.4f} ({prediction_proba[1]*100:.2f}%)")
        
        # Recommendations
        if churn_label == "Yes":
            st.markdown("### 💡 Recommended Actions")
            st.markdown("""
            - **Reach out proactively** to understand customer concerns
            - **Offer retention incentives** such as discounts or upgraded services
            - **Review contract terms** and consider offering more favorable options
            - **Improve customer service** engagement and support
            - **Analyze payment issues** if using electronic check
            """)
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Please check your input values and try again.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application predicts customer churn for a telecommunications company.
    
    **Model Information:**
    - Trained using GridSearchCV
    - Multiple algorithms compared
    - Best model selected automatically
    
    **Features Used:**
    - Demographics
    - Service subscriptions
    - Account information
    - Billing details
    
    **How to Use:**
    1. Fill in customer details
    2. Click 'Predict Churn'
    3. Review probability scores
    4. Take appropriate action
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    st.info("The model has been trained on historical customer data to provide accurate churn predictions.")
