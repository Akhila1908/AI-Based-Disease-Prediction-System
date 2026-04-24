import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

# -------------------------------------------------
# Get the directory where this script is located
# -------------------------------------------------
BASE_DIR = Path(__file__).parent

# -------------------------------------------------
# Load trained model (use relative path)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "disease_model.joblib"
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}")
        return None
    return joblib.load(model_path)

model = load_model()

# -------------------------------------------------
# Full symptom list (must match training)
# -------------------------------------------------
ALL_SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
    'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue',
    'muscle_wasting','vomiting','burning_micturition','spotting_ urination',
    'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings',
    'weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness',
    'sweating','dehydration','indigestion','headache','yellowish_skin',
    'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
    'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
    'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose',
    'congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising',
    'obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
    'enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger',
    'extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
    'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort',
    'foul_smell_of urine','continuous_feel_of_urine','passage_of_gases',
    'internal_itching','toxic_look_(typhos)','depression','irritability',
    'muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes',
    'increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma',
    'stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'blood_in_sputum','prominent_veins_on_calf','palpitations',
    'painful_walking','pus_filled_pimples','blackheads','scurring',
    'skin_peeling','silver_like_dusting','small_dents_in_nails',
    'inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

# Disease descriptions dictionary (simplified explanation)
DISEASE_INFO = {
    "Fungal infection": {"severity": "Mild to Moderate", "remedies": "Keep area dry, use antifungal cream, maintain hygiene"},
    "Allergy": {"severity": "Mild", "remedies": "Avoid allergens, take antihistamines, use cold compress"},
    "GERD": {"severity": "Moderate", "remedies": "Avoid spicy food, eat small meals, don't lie down after eating"},
    "Chronic cholestasis": {"severity": "Severe", "remedies": "Medical attention required, follow doctor's advice"},
    "Drug Reaction": {"severity": "Moderate to Severe", "remedies": "Stop suspected medication, consult doctor immediately"},
    "Peptic ulcer diseae": {"severity": "Moderate", "remedies": "Avoid NSAIDs, eat bland food, manage stress"},
    "AIDS": {"severity": "Severe", "remedies": "Medical care required, ART treatment, healthy lifestyle"},
    "Diabetes ": {"severity": "Chronic", "remedies": "Monitor blood sugar, healthy diet, regular exercise"},
    "Gastroenteritis": {"severity": "Mild to Moderate", "remedies": "Stay hydrated, rest, BRAT diet"},
    "Bronchial Asthma": {"severity": "Moderate", "remedies": "Use inhaler, avoid triggers, keep environment clean"},
    "Hypertension ": {"severity": "Chronic", "remedies": "Low salt diet, regular exercise, stress management"},
    "Migraine": {"severity": "Moderate", "remedies": "Rest in dark room, hydration, avoid triggers"},
    "Cervical spondylosis": {"severity": "Chronic", "remedies": "Neck exercises, good posture, physical therapy"},
    "Paralysis (brain hemorrhage)": {"severity": "Severe", "remedies": "Immediate medical attention, physiotherapy"},
    "Jaundice": {"severity": "Moderate", "remedies": "Rest, hydration, avoid fatty foods"},
    "Malaria": {"severity": "Severe", "remedies": "Seek medical care, anti-malarial drugs, rest"},
    "Chicken pox": {"severity": "Mild to Moderate", "remedies": "Rest, calamine lotion, avoid scratching"},
    "Dengue": {"severity": "Severe", "remedies": "Hydration, rest, medical monitoring"},
    "Typhoid": {"severity": "Severe", "remedies": "Medical care, antibiotics, rest"},
    "hepatitis A": {"severity": "Moderate", "remedies": "Rest, hydration, avoid alcohol"},
    "Hepatitis B": {"severity": "Chronic", "remedies": "Medical care, vaccination, avoid alcohol"},
    "Hepatitis C": {"severity": "Chronic", "remedies": "Medical treatment, avoid alcohol"},
    "Hepatitis D": {"severity": "Chronic", "remedies": "Medical supervision required"},
    "Hepatitis E": {"severity": "Moderate", "remedies": "Rest, hydration, medical care"},
    "Alcoholic hepatitis": {"severity": "Severe", "remedies": "Stop alcohol, medical care, nutrition"},
    "Tuberculosis": {"severity": "Severe", "remedies": "Medical treatment, complete medication course"},
    "Common Cold": {"severity": "Mild", "remedies": "Rest, fluids, honey for throat"},
    "Pneumonia": {"severity": "Severe", "remedies": "Medical care, rest, hydration"},
    "Dimorphic hemmorhoids(piles)": {"severity": "Moderate", "remedies": "Fiber-rich diet, hydration, sitz bath"},
    "Heart attack": {"severity": "Emergency", "remedies": "Call emergency, chew aspirin if advised"},
    "Varicose veins": {"severity": "Mild to Moderate", "remedies": "Exercise, compression stockings, elevate legs"},
    "Hypothyroidism": {"severity": "Chronic", "remedies": "Thyroid medication, healthy diet"},
    "Hyperthyroidism": {"severity": "Chronic", "remedies": "Medical management, avoid iodine-rich foods"},
    "Hypoglycemia": {"severity": "Moderate", "remedies": "Eat small frequent meals, monitor blood sugar"},
    "Osteoarthristis": {"severity": "Chronic", "remedies": "Exercise, weight management, pain management"},
    "Arthritis": {"severity": "Chronic", "remedies": "Exercise, anti-inflammatory diet, joint care"},
    "(vertigo) Paroymsal  Positional Vertigo": {"severity": "Moderate", "remedies": "Epley maneuver, avoid sudden movements"},
    "Acne": {"severity": "Mild", "remedies": "Gentle cleansing, avoid picking, OTC treatments"},
    "Urinary tract infection": {"severity": "Moderate", "remedies": "Hydration, cranberry juice, medical care"},
    "Psoriasis": {"severity": "Chronic", "remedies": "Moisturize, avoid triggers, medical treatment"},
    "Impetigo": {"severity": "Mild", "remedies": "Antibiotic cream, keep area clean"},
}

# -------------------------------------------------
# Convert input to vector
# -------------------------------------------------
def preprocess_symptoms(user_input):
    user_symptoms = [s.strip().lower().replace('_', ' ').replace('  ', ' ') for s in user_input.split(",")]
    # Clean symptom names
    cleaned_symptoms = []
    for s in user_symptoms:
        cleaned = s.replace(' ', '_')
        cleaned_symptoms.append(cleaned)
    
    # Create vector
    result = []
    for symptom in ALL_SYMPTOMS:
        if symptom in cleaned_symptoms:
            result.append(1)
        else:
            # Also check with space variations
            found = False
            for cs in cleaned_symptoms:
                if cs.replace('_', ' ') == symptom.replace('_', ' '):
                    found = True
                    break
            result.append(1 if found else 0)
    return result

# -------------------------------------------------
# Generate simple explanation without external API
# -------------------------------------------------
def generate_explanation(disease, symptoms):
    info = DISEASE_INFO.get(disease, {"severity": "Unknown", "remedies": "Consult a healthcare provider"})
    
    explanation = f"""
1. **What is {disease}?**
   {disease} is a medical condition that affects the body's normal functioning.

2. **Common Symptoms:**
   - Based on your input, the system analyzed your reported symptoms.

3. **Symptom Match Analysis:**
   - You reported: {symptoms}
   - This combination suggests {disease}.

4. **Severity Level:**
   🟡 **{info.get('severity', 'Unknown')}**

5. **General Home Care Suggestions:**
   • {info.get('remedies', 'Consult a healthcare provider')}

6. **When to Consult a Doctor:**
   • If symptoms persist or worsen
   • If you experience severe pain or discomfort
   • If you have fever lasting more than 3 days
   • Always consult a healthcare professional for proper diagnosis

⚠️ **Disclaimer:** This is an AI prediction tool for educational purposes only. Always consult a qualified healthcare provider for medical advice, diagnosis, or treatment.
"""
    return explanation

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Disease Prediction AI", layout="centered")

st.title("🩺 AI-Based Disease Prediction System")
st.write("Predict diseases based on symptoms and get basic information (For educational purposes only)")

# Add sidebar for info
st.sidebar.header("ℹ️ How to Use")
st.sidebar.write("""
1. Enter your symptoms separated by commas
2. Click 'Predict Disease'
3. Get the predicted disease and basic information
4. **Always consult a doctor for proper diagnosis**

**Example symptoms:**
- itching, skin_rash, nodal_skin_eruptions
- continuous_sneezing, shivering, chills
- joint_pain, stomach_pain, acidity
""")

st.sidebar.header("📋 Available Symptoms")
st.sidebar.write("Common symptoms you can try:")
common_symptoms = [
    "itching", "skin_rash", "cough", "fever", "fatigue", 
    "headache", "nausea", "vomiting", "diarrhoea", "constipation"
]
for s in common_symptoms[:10]:
    st.sidebar.write(f"- {s}")

symptoms = st.text_area(
    "Enter your symptoms (comma separated):",
    placeholder="itching, skin_rash, fatigue, headache",
    height=100
)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔍 Predict Disease", use_container_width=True):
        if symptoms.strip() == "":
            st.warning("⚠️ Please enter at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                input_data = preprocess_symptoms(symptoms)
                
                if model is None:
                    st.error("Model not loaded. Please check the model file.")
                else:
                    try:
                        predicted_disease = model.predict([input_data])[0]
                        
                        st.success(f"🧠 **Predicted Disease:** {predicted_disease}")
                        
                        # Show severity indicator
                        severity = DISEASE_INFO.get(predicted_disease, {}).get("severity", "Unknown")
                        severity_color = {
                            "Mild": "🟢",
                            "Moderate": "🟡", 
                            "Severe": "🔴",
                            "Chronic": "🟠",
                            "Emergency": "🔴🔴",
                            "Unknown": "⚪"
                        }.get(severity, "⚪")
                        
                        st.info(f"{severity_color} **Severity Level:** {severity}")
                        
                        # Generate explanation
                        explanation = generate_explanation(predicted_disease, symptoms)
                        
                        st.markdown("---")
                        st.subheader("📘 Disease Information")
                        st.markdown(explanation)
                        
                        # Add a disclaimer button
                        with st.expander("📜 Important Disclaimer"):
                            st.markdown("""
                            **This tool is for educational purposes only.**
                            
                            - This is not a substitute for professional medical advice
                            - Predictions are based on machine learning models and may not be accurate
                            - Always consult a qualified healthcare provider for diagnosis and treatment
                            - In case of emergency, contact your local emergency services immediately
                            """)
                            
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.info("Try using symptoms exactly as shown in the sidebar examples.")

# Footer
st.markdown("---")
st.caption("Built with Machine Learning | For educational purposes only")

# Run with: streamlit run ml_app.py
