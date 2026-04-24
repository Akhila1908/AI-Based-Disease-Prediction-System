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
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ============================================
# Groq LLM Setup
# ============================================

def get_groq_client():
    """Initialize Groq client - Reads API key from Streamlit secrets"""
    try:
        api_key = None
        
        # Read from Streamlit secrets (for cloud)
        try:
            if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
                if api_key:
                    print("✅ API key found in secrets")
        except Exception as e:
            print(f"Error reading secrets: {e}")
        
        if not api_key:
            return None, "GROQ_API_KEY not found in secrets"
        
        from groq import Groq
        client = Groq(api_key=api_key)
        return client, None
        
    except ImportError:
        return None, "Groq library not installed"
    except Exception as e:
        return None, str(e)

def get_disease_info_from_groq(disease_name, symptoms_list, confidence):
    """Get disease-specific information from Groq LLM"""
    
    client, error = get_groq_client()
    
    if error or client is None:
        return None
    
    prompt = f"""Provide health information for {disease_name}.

Symptoms reported: {', '.join(symptoms_list)}

Provide EXACTLY this format:

🌿 HOME REMEDIES FOR {disease_name.upper()}:
• Remedy 1
• Remedy 2
• Remedy 3
• Remedy 4
• Remedy 5

🥗 DIET RECOMMENDATIONS:
• Diet tip 1
• Diet tip 2
• Diet tip 3
• Diet tip 4
• Diet tip 5

🛡️ PREVENTION TIPS:
• Prevention tip 1
• Prevention tip 2
• Prevention tip 3
• Prevention tip 4
• Prevention tip 5

🏃‍♂️ EXERCISE GUIDELINES:
[2 sentences about safe exercises]

📚 WHEN TO SEE A DOCTOR:
[2-3 warning signs]

Keep responses specific to {disease_name}. Use simple language."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a medical information assistant. Provide specific information about {disease_name} only. Never give medical advice. Keep responses educational."
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
        st.error(f"Groq Error: {str(e)[:100]}")
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
    
    df = pd.read_csv(csv_path)
    # Remove unnamed column if exists
    df = df.drop(columns=['Unnamed: 133'], errors='ignore')
    return df

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

# Show a simple loader while loading
with st.spinner("🔄 Loading application..."):
    df = load_training_data()
    
    if df is not None:
        model, label_encoder, ALL_SYMPTOMS = get_model_and_encoder()
    else:
        model, label_encoder, ALL_SYMPTOMS = None, None, None

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
    
    # Test Groq connection
    with st.spinner("Checking Groq..."):
        client, error = get_groq_client()
        if client:
            st.success("✅ Groq AI Ready")
        else:
            st.error(f"❌ Groq: {error[:50] if error else 'Not configured'}")
            st.info("Add GROQ_API_KEY in Streamlit Secrets")
    
    st.write("---")
    st.write("### 📝 Instructions")
    st.write("1. Enter symptoms")
    st.write("2. Click Predict")
    st.write("3. Get health info")

st.write("### Enter Your Symptoms")
symptoms_input = st.text_area(
    "",
    placeholder="Example: itching, skin_rash, fatigue",
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
                
                # Get Groq information
                client, _ = get_groq_client()
                if client:
                    with st.spinner(f"🤖 Getting health information for {predicted_disease}..."):
                        info = get_disease_info_from_groq(predicted_disease, symptom_list, confidence)
                        if info:
                            st.markdown(info)
                        else:
                            st.warning("Could not fetch information. Please try again.")
                else:
                    st.info("💡 **Groq AI not available.** Add GROQ_API_KEY to secrets for home remedies and health tips.")
                
                st.markdown("---")
                st.caption("⚠️ **Educational purpose only.** Always consult a healthcare provider.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("AI Model: Random Forest | Powered by Machine Learning")
