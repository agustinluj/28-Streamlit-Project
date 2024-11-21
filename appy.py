import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# to load the model
with open(r'models\model.pkl', 'rb') as f:
    pipe = pickle.load(f)

st.title("Calories Burning Prediction")
st.write("This app predicts calories burnt during physical activities based on user inputs using a trained machine learning model.")


st.header("Enter the details:")

gender = st.selectbox("Select Gender", options=['male', 'female'], help="Choose your gender.")
age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30, help="Enter your age in years.")
height = st.number_input("Height (in cm)", min_value=50.0, max_value=250.0, value=180.0, step=0.1, help="Enter your height in centimeters.")
weight = st.number_input("Weight (in kg)", min_value=20.0, max_value=300.0, step=0.1, value=80.0, help="Enter your weight in kilograms.")
duration = st.number_input("Duration (in minutes)", min_value=1.0, max_value=300.0, step=0.1, value=45.0, help="Enter the duration of the activity in minutes.")
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, step=0.1, value=150.0, help="Enter your heart rate in beats per minute.")
body_temp = st.number_input("Body Temperature (°C)", min_value=36.0, max_value=40.0, step=0.1, value=37.0, help="Enter your body temperature in Celsius.")

# to calculate BMI and categorize it
if height > 0:
    bmi = weight / ((height / 100) ** 2)
    st.write(f"### Your BMI is: {bmi:.2f}")
    
    # categories
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal weight"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obesity"
    st.write(f"### BMI Category: {bmi_category}")



def plot_bmi_gauge(bmi_value):
    # Define the categories for BMI
    categories = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Morbidly Obese']
    colors = ['#00bfff', '#00ff00', '#ffff00', '#ffa500', '#ff0000']
    boundaries = [0, 18.5, 25, 30, 35, 50]  # BMI thresholds
    
    # Create the gauge as a pie chart
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
    start_angle = -90  # Start at the top of the gauge
    theta = np.linspace(start_angle, start_angle + 180, 100)  # Semi-circle range
    
    for i, color in enumerate(colors):
        ax.barh(1, np.deg2rad(boundaries[i + 1] - boundaries[i]) * 180 / np.pi, 
                left=np.deg2rad(boundaries[i] - 0.1), color=color, edgecolor='black', height=0.4)
                
        # Highlight the value
        ax.annotate(label,boundaries)


# to predict button
if st.button("Predict Calories Burnt"):
    # input sample 
    sample = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    })

    # prediction
    result = pipe.predict(sample)
    
    # displaying result
    st.success(f"Estimated Calories Burnt: {result[0]:.2f} kcal")

    # BMR Calculation (Harris-Benedict Equation)
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    st.write(f"### Your Basal Metabolic Rate (BMR): {bmr:.2f} kcal/day")

    # BMR chart
    fig, ax = plt.subplots(figsize=(6, 4))
    activities = ['BMR', 'Calories Burnt']
    values = [bmr, result[0]]
    ax.bar(activities, values, color=['blue', 'orange'], alpha=0.7)
    ax.set_title('BMR vs. Calories Burnt')
    ax.set_ylabel('Calories (kcal)')
    st.pyplot(fig)
