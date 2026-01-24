import streamlit as st
import joblib
import ollama
import requests

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = joblib.load("C:\\Users\\nukal\\Downloads\\disease_model.joblib")

# -------------------------------------------------
# Full symptom list (must match training)
# -------------------------------------------------
ALL_SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
    'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue',
    'muscle_wasting','vomiting','burning_micturition','spotting_urination',
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
    'foul_smell_of_urine','continuous_feel_of_urine','passage_of_gases',
    'internal_itching','toxic_look_(typhos)','depression','irritability',
    'muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic_patches','watering_from_eyes',
    'increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma',
    'stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'blood_in_sputum','prominent_veins_on_calf','palpitations',
    'painful_walking','pus_filled_pimples','blackheads','scurring',
    'skin_peeling','silver_like_dusting','small_dents_in_nails',
    'inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]
ALL_SYMPTOMS.append("unknown_symptom_placeholder")

# -------------------------------------------------
# Convert input to vector
# -------------------------------------------------
def preprocess_symptoms(user_input):
    user_symptoms = [s.strip().lower() for s in user_input.split(",")]
    return [1 if symptom in user_symptoms else 0 for symptom in ALL_SYMPTOMS]

# -------------------------------------------------
# Fetch images dynamically
# -------------------------------------------------
SERP_API_KEY = "PASTE_YOUR_KEY_HERE"

def fetch_images(query, num_images=3):
    url = "https://serpapi.com/search.json"
    params = {"q": query, "tbm": "isch", "api_key": "401fa36963fbe057b4e2204ad9b756894c4bfd67aa3d6b4b257da69734b249c7"}
    response = requests.get(url, params=params)
    data = response.json()
    return [img["original"] for img in data.get("images_results", [])[:num_images]]

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Disease Prediction AI", layout="centered")

st.title("🩺 AI-Based Disease Prediction System")
st.write("Predict diseases and get AI-based explanations (No medical advice).")

symptoms = st.text_area(
    "Enter your symptoms (comma separated):",
    placeholder="itching, skin_rash, redness"
)

if st.button("Predict Disease"):
    if symptoms.strip() == "":
        st.warning("Please enter symptoms.")
    else:
        input_data = preprocess_symptoms(symptoms)
        predicted_disease = model.predict([input_data])[0]

        st.success(f"🧠 Predicted Disease: **{predicted_disease}**")

        prompt = f"""
You are a medical awareness assistant.

Disease: {predicted_disease}
User symptoms: {symptoms}

Rules:
- Simple English only
- Bullet points only
- No technical explanations
- No diagnosis or treatment claims

Respond in this format:
1. What is the disease?
2. Common symptoms
3. Symptom match analysis
4. Severity level
5. Home remedies
6. When to consult a doctor
"""

        response = ollama.chat(
            model="phi",
            messages=[{"role": "user", "content": prompt}]
        )

        st.subheader("📘 AI Explanation")
        st.write(response["message"]["content"])

        st.subheader("🖼️ Visual References (For Awareness Only)")
        images = fetch_images(f"{predicted_disease} skin symptoms")

        for img in images:
            st.image(img, width=250)
