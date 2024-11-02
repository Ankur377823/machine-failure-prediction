import joblib
import streamlit as st
import pandas as pd
model = joblib.load("multi_output_rf_model.pkl")


type_mapping = {"L": 0, "M": 1, "H": 2}
failure_type_mapping = {
    0: "🔥 Heat Dissipation Failure",
    1: "✅ No Failure",
    2: "⚠️ Overstrain Failure",
    3: "⚡ Power Failure",
    4: "❓ Random Failures",
    5: "🛠️ Tool Wear Failure"
}

def predict_failure(input_data):
    
    prediction = model.predict(input_data)
    failure_occurrence = prediction[0][0]  
    failure_type = prediction[0][1]        

    
    if failure_occurrence == 1:
        occurrence_result = f"**Yes**, failure type: {failure_type_mapping.get(failure_type, 'Unknown Failure Type')}"
    else:
        occurrence_result = "**No**, no failure expected"
    return occurrence_result

def main():
    
    st.markdown(
        """
        <style>
        .main {
            background-color: #fef6e0; /* Cream color */
            color: #333;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🏢 Building Failure Prediction 🔍")

   
    building_type = st.selectbox("🏗️ Building Type", ["L", "M", "H"])
    air_temperature = st.number_input("🌡️ Air Temperature [K]", min_value=250.0, max_value=350.0, value=298.1)
    process_temperature = st.number_input("🌡️ Process Temperature [K]", min_value=250.0, max_value=400.0, value=308.6)
    rotational_speed = st.number_input("🔄 Rotational Speed [rpm]", min_value=0, max_value=5000, value=1551)
    torque = st.number_input("🔧 Torque [Nm]", min_value=0.0, max_value=500.0, value=42.8)
    tool_wear = st.number_input("⏳ Tool Wear [min]", min_value=0, max_value=1000, value=0)

    
    encoded_building_type = type_mapping[building_type]

    
    input_data = pd.DataFrame([{
        'Type': encoded_building_type,
        'Air temperature [K]': air_temperature,
        'Process temperature [K]': process_temperature,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }])

    
    if st.button("🔮 Predict"):
        try:
            fail_result = predict_failure(input_data)
            st.write("**Failure Occurrence and Type Prediction:**", fail_result)
        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
