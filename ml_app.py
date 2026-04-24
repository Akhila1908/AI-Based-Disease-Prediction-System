# ============================================
# MUST BE FIRST - Streamlit page config
# ============================================
import streamlit as st
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ============================================
# Imports
# ============================================
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ============================================
# Groq LLM Setup
# ============================================

def get_groq_client():
    """Initialize Groq client"""
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            return None, "GROQ_API_KEY not found"
        
        from groq import Groq
        client = Groq(api_key=api_key)
        return client, None
        
    except Exception as e:
        return None, str(e)

def get_disease_info_from_groq(disease_name, symptoms_list, confidence):
    """Get disease-specific information from Groq LLM"""
    
    client, error = get_groq_client()
    
    if error or client is None:
        return None
    
    prompt = f"""Provide health information for {disease_name} based on these symptoms: {', '.join(symptoms_list)} (confidence: {confidence:.1f}%)

Provide ONLY in this exact format:

🌿 HOME REMEDIES:
• [remedy 1]
• [remedy 2]
• [remedy 3]
• [remedy 4]
• [remedy 5]

🥗 DIET RECOMMENDATIONS:
• [diet 1]
• [diet 2]
• [diet 3]
• [diet 4]
• [diet 5]

🛡️ PREVENTION TIPS:
• [tip 1]
• [tip 2]
• [tip 3]
• [tip 4]
• [tip 5]

🏃‍♂️ EXERCISE GUIDELINES:
[2 sentences about exercise]

📚 WHEN TO SEE A DOCTOR:
[2-3 warning signs]

Keep responses specific to {disease_name}. Use practical, actionable advice."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a medical information assistant. Provide specific, practical information about {disease_name}. Never give medical advice. Always recommend consulting doctors."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        return None

# ============================================
# Load data from CSV only
# ============================================
@st.cache_data
def load_training_data():
    csv_path = BASE_DIR / 'Training.csv'
    if not csv_path.exists():
        st.error("❌ Training.csv not found!")
        return None
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 133'], errors='ignore')
    return df

@st.cache_data
def get_symptom_list(df):
    if df is None:
        return []
    return [col for col in df.columns if col != 'prognosis']

@st.cache_data
def get_disease_list(df):
    if df is None:
        return []
    return sorted(df['prognosis'].unique())

@st.cache_resource
def load_or_train_model(df):
    model_path = BASE_DIR / "disease_model.joblib"
    encoder_path = BASE_DIR / "label_encoder.joblib"
    
    if df is None:
        return None, None
    
    if model_path.exists() and encoder_path.exists():
        try:
            model = joblib.load(model_path)
            le = joblib.load(encoder_path)
            return model, le
        except Exception as e:
            st.warning(f"Could not load model: {e}")
    
    with st.spinner("🔄 Training AI model from dataset..."):
        try:
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
                ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
            ])
            
            pipeline.fit(X, y_encoded)
            
            joblib.dump(pipeline, model_path)
            joblib.dump(le, encoder_path)
            
            return pipeline, le
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return None, None

def preprocess_symptoms(user_input, all_symptoms):
    if not user_input.strip() or not all_symptoms:
        return [0] * 132
    
    user_symptoms = [s.strip().lower().replace(' ', '_') for s in user_input.split(",")]
    
    result = []
    for symptom in all_symptoms:
        clean_symptom = symptom.strip().lower()
        matched = any(us == clean_symptom or us.replace('_', ' ') == clean_symptom.replace('_', ' ') 
                     for us in user_symptoms)
        result.append(1 if matched else 0)
    return result

# ============================================
# Load data
# ============================================
df = load_training_data()

if df is None:
    st.stop()

ALL_SYMPTOMS = get_symptom_list(df)
ALL_DISEASES = get_disease_list(df)
model, label_encoder = load_or_train_model(df)

# Check Groq availability
client, _ = get_groq_client()
GROQ_AVAILABLE = client is not None

# ============================================
# UI
# ============================================
st.title("🩺 AI Disease Prediction System")

# Sidebar
with st.sidebar:
    st.write("### 📊 Dataset Info")
    st.write(f"**Diseases:** {len(ALL_DISEASES)}")
    st.write(f"**Symptoms:** {len(ALL_SYMPTOMS)}")
    st.write(f"**Training Records:** {len(df)}")
    
    st.write("---")
    if GROQ_AVAILABLE:
        st.success("✅ Groq AI Ready")
        st.caption("Will provide disease-specific information")
    else:
        st.warning("⚠️ Groq AI not configured")
        st.caption("Add GROQ_API_KEY to secrets for health info")
    
    st.write("---")
    st.write("### 📝 How to Use")
    st.write("1. Enter symptoms (comma separated)")
    st.write("2. Click Predict")
    st.write("3. Get disease-specific health info")

# Main input
symptoms_input = st.text_area(
    "**Enter your symptoms (comma separated):**",
    placeholder="Example: itching, skin_rash, fatigue, headache",
    height=100
)

if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
    if not symptoms_input.strip():
        st.warning("⚠️ Please enter at least one symptom")
    elif model is None:
        st.error("❌ Model not available")
    else:
        with st.spinner("🧠 Analyzing..."):
            try:
                # Get prediction
                input_vector = preprocess_symptoms(symptoms_input, ALL_SYMPTOMS)
                pred_encoded = model.predict([input_vector])[0]
                predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]
                
                probs = model.predict_proba([input_vector])[0]
                confidence = max(probs) * 100
                
                symptom_list = [s.strip() for s in symptoms_input.split(",") if s.strip()]
                
                # Display prediction
                st.success(f"### 🎯 Predicted: {predicted_disease}")
                st.metric("Confidence", f"{confidence:.1f}%")
                st.write(f"**Symptoms reported:** {', '.join(symptom_list)}")
                
                st.markdown("---")
                
                # Get disease-specific information from Groq ONLY
                if GROQ_AVAILABLE:
                    with st.spinner(f"🤖 Getting information about {predicted_disease}..."):
                        disease_info = get_disease_info_from_groq(predicted_disease, symptom_list, confidence)
                        
                        if disease_info:
                            st.markdown(disease_info)
                        else:
                            st.error(f"Could not fetch information for {predicted_disease}. Please try again.")
                else:
                    st.info("💡 **Groq AI not configured.** Add GROQ_API_KEY to Streamlit secrets to get home remedies, diet tips, and prevention advice.")
                    
                    # Show basic info from CSV only
                    disease_data = df[df['prognosis'] == predicted_disease]
                    common_symptoms = []
                    for sym in ALL_SYMPTOMS[:20]:
                        if len(disease_data) > 0 and disease_data[sym].mean() > 0.5:
                            common_symptoms.append(sym.replace('_', ' ').title())
                    
                    if common_symptoms:
                        st.write("**Common symptoms for this condition (from training data):**")
                        cols = st.columns(3)
                        for i, sym in enumerate(common_symptoms[:9]):
                            with cols[i % 3]:
                                st.write(f"- {sym}")
                
                # Disclaimer
                st.markdown("---")
                st.warning("""
                ⚠️ **Medical Disclaimer:** Educational purpose only.  
                Always consult a healthcare provider for medical advice.
                """)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("AI Model: Random Forest | Powered by Machine Learning")
