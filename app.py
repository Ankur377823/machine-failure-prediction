import os
import joblib
import streamlit as st
import pandas as pd

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "machine_failure.pkl")
model = joblib.load(MODEL_PATH)

type_mapping = {"L": 0, "M": 1, "H": 2}

failure_type_mapping = {
    0: "🔥 Heat Dissipation Failure",
    1: "⚠️ Overstrain Failure",
    2: "⚡ Power Failure",
    3: "❓ Random Failure",
    4: "🛠️ Tool Wear Failure"
}

def main():
    st.title("⚠️ Machine Failure Prediction 🛠️")

    machine_type = st.selectbox("🏗️ Machine Quality", ["L", "M", "H"])
    air_temperature = st.number_input("🌡️ Air Temperature (K)", 250.0, 350.0, 298.1)
    process_temperature = st.number_input("🌡️ Process Temperature (K)", 250.0, 400.0, 308.6)
    rotational_speed = st.number_input("🔄 Rotational Speed (rpm)", 0, 5000, 1551)
    torque = st.number_input("🔧 Torque (Nm)", 0.0, 500.0, 42.8)
    tool_wear = st.number_input("⏳ Tool Wear (min)", 0, 1000, 0)

    input_data = pd.DataFrame([{
        "Type": type_mapping[machine_type],
        "Air temperature K": air_temperature,
        "Process temperature K": process_temperature,
        "Rotational speed rpm": rotational_speed,
        "Torque Nm": torque,
        "Tool wear min": tool_wear
    }])

    if st.button("🔮 Predict"):
        try:
            pred = model.predict(input_data)[0]

            if model.n_classes_ == 2:
                prob = model.predict_proba(input_data)[0][1]

                if pred == 1:
                    st.error(f"❌ Failure Expected\n\n📊 Probability: {prob:.2f}")
                else:
                    st.success(f"✅ No Failure Expected\n\n📊 Probability: {prob:.2f}")
            else:
                st.error(
                    f"❌ Failure Detected\n\n"
                    f"🔍 Type: {failure_type_mapping.get(pred, 'Unknown')}"
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
