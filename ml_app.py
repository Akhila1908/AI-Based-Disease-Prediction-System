# ============================================
# MUST BE FIRST - Streamlit page config
# ============================================
import streamlit as st
st.set_page_config(page_title="Disease Prediction AI", layout="centered")

# ============================================
# Now import other libraries
# ============================================
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ============================================
# Full symptom list (132 symptoms)
# ============================================
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

# ============================================
# Disease information
# ============================================
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

# ============================================
# Load or train model
# ============================================
@st.cache_resource
def load_or_train_model():
    """Load model if exists, otherwise train it"""
    model_path = BASE_DIR / "disease_model.joblib"
    encoder_path = BASE_DIR / "label_encoder.joblib"
    
    # Try to load existing model
    if model_path.exists() and encoder_path.exists():
        try:
            model = joblib.load(model_path)
            le = joblib.load(encoder_path)
            return model, le
        except Exception as e:
            st.warning(f"Could not load existing model: {e}")
    
    # Train new model (only if Training.csv exists)
    csv_path = BASE_DIR / 'Training.csv'
    if not csv_path.exists():
        st.error("❌ Training.csv not found! Please upload the dataset.")
        return None, None
    
    with st.spinner("🔄 Training model (this may take 1-2 minutes)..."):
        try:
            # Load data
            df = pd.read_csv(csv_path)
            df = df.drop(columns=['Unnamed: 133'], errors='ignore')
            
            X = df.drop('prognosis', axis=1)
            y = df['prognosis']
            
            from sklearn.preprocessing import LabelEncoder
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
            ])
            
            pipeline.fit(X, y_encoded)
            
            # Save models
            joblib.dump(pipeline, model_path)
            joblib.dump(le, encoder_path)
            
            st.success("✅ Model trained successfully!")
            return pipeline, le
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return None, None

# ============================================
# Helper functions
# ============================================
def preprocess_symptoms(user_input):
    """Convert user input to feature vector"""
    if not user_input.strip():
        return [0] * len(ALL_SYMPTOMS)
    
    user_symptoms = [s.strip().lower().replace(' ', '_') for s in user_input.split(",")]
    result = []
    for symptom in ALL_SYMPTOMS:
        matched = any(
            us == symptom or us.replace('_', ' ') == symptom.replace('_', ' ')
            for us in user_symptoms
        )
        result.append(1 if matched else 0)
    return result

def generate_explanation(disease, symptoms):
    """Generate explanation for the predicted disease"""
    info = DISEASE_INFO.get(disease, {"severity": "Unknown", "remedies": "Consult a healthcare provider"})
    
    explanation = f"""
**1. What is {disease}?**  
{disease} is a medical condition that affects the body's normal functioning.

**2. Symptom Match Analysis:**  
You reported: {symptoms}  
This combination suggests {disease}.

**3. Severity Level:**  
🟡 **{info.get('severity', 'Unknown')}**

**4. General Home Care Suggestions:**  
• {info.get('remedies', 'Consult a healthcare provider')}

**5. When to Consult a Doctor:**  
• If symptoms persist or worsen  
• If you experience severe pain or discomfort  
• If you have fever lasting more than 3 days  

⚠️ **Disclaimer:** This is an AI prediction tool for educational purposes only. Always consult a qualified healthcare provider.
"""
    return explanation

# ============================================
# Load model (this runs after page config)
# ============================================
model, label_encoder = load_or_train_model()

# ============================================
# Streamlit UI
# ============================================
st.title("🩺 AI-Based Disease Prediction System")
st.write("Predict diseases based on symptoms (For educational purposes only)")

st.sidebar.header("ℹ️ How to Use")
st.sidebar.write("""
1. Enter symptoms separated by commas
2. Click 'Predict Disease'
3. Get the predicted disease

**Example:** itching, skin_rash, fatigue

**Note:** Use symptom names as shown in the dataset.
""")

if model is None or label_encoder is None:
    st.error("❌ Model not available. Please check that Training.csv is in the repository.")
    st.stop()

symptoms = st.text_area(
    "Enter your symptoms (comma separated):",
    placeholder="itching, skin_rash, fatigue, headache",
    height=100
)

if st.button("🔍 Predict Disease", type="primary"):
    if symptoms.strip() == "":
        st.warning("⚠️ Please enter at least one symptom.")
    else:
        with st.spinner("Analyzing symptoms..."):
            try:
                input_vector = preprocess_symptoms(symptoms)
                
                # Make prediction
                prediction_encoded = model.predict([input_vector])[0]
                
                # Convert to disease name
                predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]
                
                st.success(f"🧠 **Predicted Disease:** {predicted_disease}")
                
                # Generate and show explanation
                explanation = generate_explanation(predicted_disease, symptoms)
                st.markdown("---")
                st.markdown(explanation)
                
                with st.expander("📜 Important Disclaimer"):
                    st.markdown("""
                    **This tool is for educational purposes only.**
                    
                    - Not a substitute for professional medical advice
                    - Always consult a qualified healthcare provider
                    - In case of emergency, contact local emergency services
                    """)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

st.markdown("---")
st.caption("Built with Machine Learning | Educational purposes only")
