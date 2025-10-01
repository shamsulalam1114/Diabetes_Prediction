import streamlit as st
import joblib
import pandas as pd
from fpdf import FPDF
from datetime import datetime


# --- Clinical Interpretation Function ---
def get_clinical_interpretation(patient_data):
    """Generates clinical interpretation based on WHO guidelines and feature thresholds."""
    interpretations = []
    risk_factors = []

    # 1. Glucose
    glucose = patient_data['glucose']
    if glucose >= 126:
        interpretations.append(
            f"**Glucose Level ({glucose} mg/dL):** This level is in the diabetic range. It is highly recommended to consult a healthcare professional for confirmation and management.")
        risk_factors.append("High Glucose Level")
    elif 100 <= glucose <= 125:
        interpretations.append(
            f"**Glucose Level ({glucose} mg/dL):** This is in the prediabetic range. Lifestyle modifications, including diet and exercise, are advised to prevent progression to diabetes.")
        risk_factors.append("Prediabetic Glucose Level")
    else:
        interpretations.append(
            f"**Glucose Level ({glucose} mg/dL):** This is within the normal range. Maintain a healthy lifestyle.")

    # 2. BMI
    bmi = patient_data['bmi']
    if bmi >= 30:
        interpretations.append(
            f"**BMI ({bmi:.2f} kg/m²):** This is in the obese range, which is a significant risk factor for Type 2 Diabetes. Weight management is strongly advised.")
        risk_factors.append("Obesity (High BMI)")
    elif 25 <= bmi < 30:
        interpretations.append(
            f"**BMI ({bmi:.2f} kg/m²):** This is in the overweight range. A healthy diet and regular physical activity are recommended to reduce diabetes risk.")
        risk_factors.append("Overweight")
    else:
        interpretations.append(
            f"**BMI ({bmi:.2f} kg/m²):** The patient's BMI is in the normal range. This is excellent for metabolic health.")

    # 3. Blood Pressure
    systolic = patient_data['systolic_bp']
    diastolic = patient_data['diastolic_bp']
    if systolic >= 140 or diastolic >= 90:
        interpretations.append(
            f"**Blood Pressure ({systolic}/{diastolic} mmHg):** This indicates Hypertension (Stage 2), a major risk factor for cardiovascular diseases and associated with diabetes. Medical consultation is essential.")
        risk_factors.append("High Blood Pressure")
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        interpretations.append(
            f"**Blood Pressure ({systolic}/{diastolic} mmHg):** This indicates Hypertension (Stage 1). Monitoring and lifestyle changes are important.")
        risk_factors.append("Elevated Blood Pressure")
    else:
        interpretations.append(
            f"**Blood Pressure ({systolic}/{diastolic} mmHg):** Blood pressure is within the normal range.")

    # 4. Age
    age = patient_data['age']
    if age >= 45:
        interpretations.append(
            f"**Age ({age} years):** Being over 45 increases the risk for Type 2 Diabetes. Regular check-ups are recommended.")
        risk_factors.append("Age over 45")

    # 5. Hypertensive
    if patient_data['hypertensive'] == 1:
        risk_factors.append("Existing Hypertension Diagnosis")

    if not risk_factors:
        risk_factors.append("No major risk factors identified from the provided data.")

    return interpretations, risk_factors


# --- PDF Report Generation Function ---
def create_pdf_report(patient_data, prediction_result, confidence, interpretations, risk_factors):
    """Generates a PDF report with patient data, prediction, and clinical interpretation."""
    pdf = FPDF()
    pdf.add_page()

    # Calculate effective page width
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    # Title
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "Diabetes Prediction Report", 0, 1, 'C')
    pdf.ln(10)

    # Report Info
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)

    # Patient Data Section
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Patient Information", 0, 1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    for key, value in patient_data.items():
        display_key = str(key).replace('_', ' ').title()
        pdf.cell(95, 10, f"{display_key}:", 0, 0)
        pdf.cell(95, 10, str(value), 0, 1)
    pdf.ln(10)

    # Prediction Result Section
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Prediction Result", 0, 1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    if prediction_result == "Positive":
        pdf.set_text_color(220, 53, 69)
        result_text = "High Likelihood of Diabetes"
    else:
        pdf.set_text_color(40, 167, 69)
        result_text = "Low Likelihood of Diabetes"
    pdf.cell(0, 10, result_text, 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Model Confidence: {confidence}", 0, 1)
    pdf.ln(5)

    # Clinical Interpretation Section
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Clinical Interpretation", 0, 1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    for line in interpretations:
        pdf.multi_cell(effective_width, 5, line.replace("**", ""), border=0, align='L')
        pdf.ln(2)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Primary Risk Factors Identified:", 0, 1)
    pdf.set_font("Arial", '', 12)
    for factor in risk_factors:
        pdf.multi_cell(effective_width, 5, f"- {factor}", border=0, align='L')

    pdf.ln(15)

    # Disclaimer
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(effective_width, 5,
                   "Disclaimer: This is a prediction generated by a machine learning model and should not be considered a substitute for a professional medical diagnosis. Please consult a healthcare provider for any health concerns.",
                   0, 'C')

    return bytes(pdf.output())


# --- Load Model ---
try:
    model = joblib.load("best_global_stacking_model.pkl")
except FileNotFoundError:
    st.error("Error: The model file 'best_global_stacking_model.pkl' was not found.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop

# --- Page Config & Styling ---
st.set_page_config(page_title="DiaPredict AI", layout="wide")
st.markdown(
    """
    <style>
    .main, .stApp { background-color: #f0f2f6; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    body { color: #31333F; }
    label { color: #31333F !important; }
    h1, h3 { color: #1c4e80 !important; text-align: center; }
    [data-testid="stMetricLabel"] { color: #5a5a5a !important; }
    [data-testid="stMetricValue"] { color: #31333F !important; }
    .stButton>button {
        width: 100%; border-radius: 20px; border: 1px solid #1c4e80;
        background-color: #1c4e80; color: white; transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { background-color: white; color: #1c4e80; }
    .stButton>button:active { background-color: #153b66; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Layout ---
st.title("🩺 DiaPredict AI")
st.markdown(
    "<h4 style='text-align: center; color: #5a5a5a;'>Your AI-powered assistant for diabetes risk assessment.</h4>",
    unsafe_allow_html=True)
st.markdown("---")

# --- Centered Content ---
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    with st.container(border=True):
        st.subheader("Patient Health Metrics")
        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
                pulse_rate = st.number_input("Pulse Rate (per min)", min_value=30, max_value=200, value=72)
                systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120)
                diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=150, value=80)
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=500, value=100)
            with col2:
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
                weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
                bmi = st.number_input("Body Mass Index (kg/m²)", min_value=10.0, max_value=60.0, value=24.0,
                                      format="%.2f")
                hypertensive = st.selectbox("Patient is Hypertensive?", ["No", "Yes"])
                diagnostic_label = st.selectbox("Diagnostic Label", ["Normal", "Prediabetes", "Diabetes"])
            submitted = st.form_submit_button("Submit for Prediction")

    # --- Prediction and Report Display ---
    st.markdown("---")
    if submitted:
        # Map inputs to numerical values for the model
        hypertensive_val = 1 if hypertensive == "Yes" else 0
        diagnostic_label_map = {"Normal": 0, "Prediabetes": 1, "Diabetes": 2}
        diagnostic_label_val = diagnostic_label_map[diagnostic_label]

        input_data_dict = {
            'age': age, 'pulse_rate': pulse_rate, 'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp, 'glucose': glucose, 'height': height,
            'weight': weight, 'bmi': bmi, 'hypertensive': hypertensive_val,
            'diagnostic_label': diagnostic_label_val
        }
        input_df = pd.DataFrame([input_data_dict])

        try:
            prediction = model.predict(input_df)[0]
            if hasattr(model, "predict_proba"):
                positive_class_proba = model.predict_proba(input_df)[0][1]
            else:
                positive_class_proba = None

            with st.container(border=True):
                st.subheader("Prediction Outcome")
                if prediction == 1:
                    st.error("Result: High Likelihood of Diabetes", icon="⚠️")
                    confidence_score = f"{positive_class_proba:.2%}" if positive_class_proba is not None else "N/A"
                    prediction_text = "Positive"
                else:
                    st.success("Result: Low Likelihood of Diabetes", icon="✅")
                    confidence_score = f"{1 - positive_class_proba:.2%}" if positive_class_proba is not None else "N/A"
                    prediction_text = "Negative"

                st.metric(label="Prediction Confidence", value=confidence_score)
                st.caption("This confidence score represents the model's certainty in its prediction.")

                # --- Display Clinical Interpretation ---
                st.markdown("---")
                st.subheader("Clinical Interpretation & Risk Assessment")
                interpretations, risk_factors = get_clinical_interpretation(input_data_dict)
                for line in interpretations:
                    st.markdown(line, unsafe_allow_html=True)

                st.markdown("**Primary Risk Factors Identified:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")

                # Generate and offer the PDF for download
                pdf_bytes = create_pdf_report(input_data_dict, prediction_text, confidence_score, interpretations,
                                              risk_factors)
                st.download_button(
                    label="📥 Download Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"diabetes_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.info("The prediction result will be displayed here after you submit the patient's data.")

