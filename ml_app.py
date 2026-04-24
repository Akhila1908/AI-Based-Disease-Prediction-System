# ============================================
# MUST BE FIRST - Streamlit page config
# ============================================
import streamlit as st
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ============================================
# Imports
# ============================================
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import requests
import json
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ============================================
# Ollama Setup - No API key needed!
# ============================================

def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_disease_info_from_ollama(disease_name, symptoms_list, confidence):
    """Get disease-specific information using Ollama"""
    
    prompt = f"""You are a helpful medical assistant. Provide health information for {disease_name}.

Symptoms reported: {', '.join(symptoms_list)}
Confidence: {confidence:.1f}%

Provide EXACTLY this format (use simple language, no markdown):

HOME REMEDIES:
1. [specific remedy 1]
2. [specific remedy 2]
3. [specific remedy 3]
4. [specific remedy 4]
5. [specific remedy 5]

DIET RECOMMENDATIONS:
1. [diet tip 1]
2. [diet tip 2]
3. [diet tip 3]
4. [diet tip 4]
5. [diet tip 5]

PREVENTION TIPS:
1. [prevention tip 1]
2. [prevention tip 2]
3. [prevention tip 3]
4. [prevention tip 4]
5. [prevention tip 5]

WHEN TO SEE A DOCTOR:
- [warning sign 1]
- [warning sign 2]
- [warning sign 3]

Keep responses practical and specific to {disease_name}. Always recommend consulting a doctor."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",  # or "mistral", "llama2", "tinyllama"
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 800
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response from Ollama")
        else:
            return None
            
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"Ollama Error: {str(e)[:100]}")
        return None

# ============================================
# Load data and train model
# ============================================
@st.cache_data
def load_training_data():
    csv_path = BASE_DIR / 'Training.csv'
    if not csv_path.exists():
        st.error("❌ Training.csv not found!")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df = df.drop(columns=['Unnamed: 133'], errors='ignore')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

@st.cache_resource
def get_model_and_encoder():
    """Load or train model"""
    df = load_training_data()
    if df is None:
        return None, None, None
    
    # Get symptom list
    symptom_columns = [col for col in df.columns if col != 'prognosis']
    
    # Train model
    try:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        X = df.drop('prognosis', axis=1)
        y = df['prognosis']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X, y_encoded)
        
        return pipeline, le, symptom_columns
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None, None, None

def preprocess_symptoms(user_input, all_symptoms):
    if not user_input.strip() or not all_symptoms:
        return [0] * len(all_symptoms) if all_symptoms else [0] * 132
    
    user_symptoms = [s.strip().lower().replace(' ', '_') for s in user_input.split(",")]
    
    result = []
    for symptom in all_symptoms:
        clean_symptom = symptom.strip().lower()
        matched = any(us == clean_symptom or us.replace('_', ' ') == clean_symptom.replace('_', ' ') 
                     for us in user_symptoms)
        result.append(1 if matched else 0)
    return result

# ============================================
# Load everything
# ============================================

with st.spinner("🔄 Loading application..."):
    df = load_training_data()
    
    if df is not None:
        model, label_encoder, ALL_SYMPTOMS = get_model_and_encoder()
    else:
        model, label_encoder, ALL_SYMPTOMS = None, None, None

# Check Ollama status
ollama_available = check_ollama()

# ============================================
# UI
# ============================================
st.title("🩺 AI Disease Prediction System")

with st.sidebar:
    st.write("### 📊 Status")
    if df is not None:
        st.success(f"✅ Dataset loaded: {len(df)} records")
        st.write(f"**Diseases:** {df['prognosis'].nunique()}")
        st.write(f"**Symptoms:** {len(ALL_SYMPTOMS)}")
    else:
        st.error("❌ Dataset not found")
    
    st.write("---")
    
    # Check Ollama status
    if ollama_available:
        st.success("✅ Ollama AI Ready")
        st.caption("Model: phi (for health information)")
    else:
        st.warning("⚠️ Ollama not available")
        st.caption("Will show basic info from training data")
    
    st.write("---")
    st.write("### 📝 Instructions")
    st.write("1. Enter symptoms (comma separated)")
    st.write("2. Click Predict")
    st.write("3. Get health information")

st.write("### Enter Your Symptoms")
symptoms_input = st.text_area(
    "",
    placeholder="Example: itching, skin_rash, fatigue, headache",
    height=80,
    label_visibility="collapsed"
)

# Center the button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("🔍 Predict Disease", type="primary", use_container_width=True)

if predict_clicked:
    if not symptoms_input.strip():
        st.warning("⚠️ Please enter at least one symptom")
    elif model is None:
        st.error("❌ Model not ready. Please wait or refresh.")
    else:
        with st.spinner("🧠 Analyzing your symptoms..."):
            try:
                # Preprocess and predict
                input_vector = preprocess_symptoms(symptoms_input, ALL_SYMPTOMS)
                pred_encoded = model.predict([input_vector])[0]
                predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]
                
                # Get confidence
                probs = model.predict_proba([input_vector])[0]
                confidence = max(probs) * 100
                
                symptom_list = [s.strip() for s in symptoms_input.split(",") if s.strip()]
                
                # Display result
                st.success(f"### 🎯 Predicted: {predicted_disease}")
                
                # Show confidence with color
                if confidence >= 80:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="High")
                elif confidence >= 60:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="Medium")
                else:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="Low")
                
                st.markdown("---")
                
                # Get Ollama information if available
                if ollama_available:
                    with st.spinner(f"🤖 Getting health information for {predicted_disease}..."):
                        info = get_disease_info_from_ollama(predicted_disease, symptom_list, confidence)
                        if info:
                            st.markdown(info)
                        else:
                            st.warning("Could not fetch information from Ollama. Make sure Ollama is running.")
                            # Show fallback info from training data
                            disease_data = df[df['prognosis'] == predicted_disease]
                            common_symptoms = []
                            for sym in ALL_SYMPTOMS[:20]:
                                if len(disease_data) > 0 and disease_data[sym].mean() > 0.5:
                                    common_symptoms.append(sym.replace('_', ' ').title())
                            
                            if common_symptoms:
                                st.write("**Common symptoms (from training data):**")
                                cols = st.columns(3)
                                for i, sym in enumerate(common_symptoms[:9]):
                                    with cols[i % 3]:
                                        st.write(f"- {sym}")
                else:
                    st.info("💡 **Ollama not available.** For home remedies and health tips, install Ollama locally or add GROQ_API_KEY to secrets.")
                    
                    # Show common symptoms from training data as fallback
                    disease_data = df[df['prognosis'] == predicted_disease]
                    common_symptoms = []
                    for sym in ALL_SYMPTOMS[:20]:
                        if len(disease_data) > 0 and disease_data[sym].mean() > 0.5:
                            common_symptoms.append(sym.replace('_', ' ').title())
                    
                    if common_symptoms:
                        st.write("**Common symptoms (from training data):**")
                        cols = st.columns(3)
                        for i, sym in enumerate(common_symptoms[:9]):
                            with cols[i % 3]:
                                st.write(f"- {sym}")
                
                st.markdown("---")
                st.caption("⚠️ **Educational purpose only.** Always consult a healthcare provider.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("AI Model: Random Forest | Powered by Machine Learning")
