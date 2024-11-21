import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# to load the model
with open('model.pkl', 'rb') as f:
    pipe = pickle.load(f)

st.title("Calories Burning Prediction")
st.write("This app predicts calories burnt during physical activities based on user inputs using a trained machine learning model.")


st.header("Enter the details:")

gender = st.selectbox("Select Gender", options=['male', 'female'], help="Choose your gender.")
age = st.number_input("Age", min_value=1, max_value=120, step=1, help="Enter your age in years.")
height = st.number_input("Height (in cm)", min_value=50.0, max_value=250.0, step=0.1, help="Enter your height in centimeters.")
weight = st.number_input("Weight (in kg)", min_value=20.0, max_value=300.0, step=0.1, help="Enter your weight in kilograms.")
duration = st.number_input("Duration (in minutes)", min_value=1.0, max_value=300.0, step=0.1, help="Enter the duration of the activity in minutes.")
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, step=0.1, help="Enter your heart rate in beats per minute.")
body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1, help="Enter your body temperature in Celsius.")

# BMI Calculation and Categorization to calculate BMI and
if height > 0:
    bmi = weight / ((height / 100) ** 2)
    st.write(f"### Your BMI is: {bmi:.2f}")
    
    # BMI categories
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal weight"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obesity"
    st.write(f"### BMI Category: {bmi_category}")

    # BMI plot
    fig, ax = plt.subplots()
    categories = ["Underweight", "Normal weight", "Overweight", "Obesity"]
    thresholds = [18.5, 24.9, 29.9, 40]  # BMI thresholds
    colors = ['blue', 'green', 'orange', 'red']
    bars = ax.bar(categories, thresholds, color=colors, alpha=0.7)

    # Highlight user's BMI
    for bar, category in zip(bars, categories):
        if category == bmi_category:
            bar.set_color('yellow')
            bar.set_edgecolor('black')
            bar.set_linewidth(2)

    ax.axhline(bmi, color='black', linestyle='--', label=f"Your BMI: {bmi:.2f}")
    ax.legend()
    ax.set_title("BMI Classification")
    ax.set_ylabel("BMI Thresholds")
    st.pyplot(fig)

# Predict button
if st.button("Predict Calories Burnt"):
    # Create input sample
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
    
    # display result
    st.success(f"Estimated Calories Burnt: {result[0]:.2f} kcal")

    # relationship between Duration and Predicted Calories
    st.header("Visualization of Calories Burnt vs Duration")
    durations = range(1, 301, 10)
    calories = [pipe.predict(pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [d],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    }))[0] for d in durations]

    # Plot
    fig, ax = plt.subplots()
    ax.plot(durations, calories, label="Predicted Calories Burnt", color='orange')
    ax.scatter([duration], [result[0]], color='red', label="Your Input", zorder=5)
    ax.set_title("Calories Burnt vs Duration")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Calories Burnt (kcal)")
    ax.legend()
    st.pyplot(fig)
