
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("Wellness Tourism Package Prediction")
st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to purchase the Wellness Tourism Package."
)

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

MODEL_PATH = "tourism_project/deployment/tourism_model.joblib"

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------
# Customer input form
# ---------------------------------------------------------

st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    city_tier = st.selectbox(
        "City Tier",
        options=[1, 2, 3],
        index=2
    )

    type_of_contact = st.selectbox(
        "Type of Contact",
        options=["Self Enquiry", "Company Invited"]
    )

    occupation = st.selectbox(
        "Occupation",
        options=[
            "Salaried",
            "Free Lancer",
            "Small Business",
            "Large Business"
        ]
    )

    gender = st.selectbox(
        "Gender",
        options=["Female", "Male", "Fe Male"]
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=10,
        value=3
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=1,
        max_value=200,
        value=15
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        options=[3, 4, 5],
        index=1
    )

with col2:

    product_pitched = st.selectbox(
        "Product Pitched",
        options=[
            "Basic",
            "Deluxe",
            "Standard",
            "Super Deluxe",
            "King"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        options=[
            "Single",
            "Divorced",
            "Married",
            "Unmarried"
        ]
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=30,
        value=3
    )

    passport = st.selectbox(
        "Passport",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    pitch_satisfaction_score = st.selectbox(
        "Pitch Satisfaction Score",
        options=[1, 2, 3, 4, 5],
        index=2
    )

    own_car = st.selectbox(
        "Own Car",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=1
    )

    designation = st.selectbox(
        "Designation",
        options=[
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=200000,
        value=25000
    )

# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The customer is predicted to purchase the "
            "Wellness Tourism Package."
        )
    else:
        st.info(
            "The customer is predicted not to purchase the "
            "Wellness Tourism Package."
        )

    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )
