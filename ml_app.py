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
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

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
# Groq LLM with direct API call
# ============================================

def get_health_info_from_groq(disease_name, symptoms_list, confidence):
    """Get health information using direct Groq API call with better formatting"""
    
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except:
        pass
    
    if not api_key:
        return None
    
    import requests
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are a medical information assistant. Provide detailed health information for {disease_name}.

User symptoms: {', '.join(symptoms_list)}
Confidence: {confidence:.1f}%

Provide the response in EXACTLY this format with clear sections and bullet points:

**About the condition**
[1-2 sentences explaining what {disease_name} is]

**Common symptoms**
- Symptom 1
- Symptom 2
- Symptom 3
- Symptom 4
- Symptom 5

**Do the given symptoms match?**
[Yes/No] - [1 sentence explanation]

**Is it mild or serious?**
[1 sentence explaining severity]

**General home care**
- Home care tip 1
- Home care tip 2
- Home care tip 3
- Home care tip 4
- Home care tip 5

**When to consult a doctor**
- Warning sign 1
- Warning sign 2
- Warning sign 3

Keep responses practical, specific to {disease_name}, and easy to read. Use simple language. Never give medical advice."""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": f"You are a medical information assistant. Provide clear, organized health information about {disease_name}. Use bullet points with dashes. Never give medical advice."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            return content
        else:
            return None
            
    except Exception as e:
        return None

# ============================================
# Load everything
# ============================================

with st.spinner("🔄 Loading application..."):
    df = load_training_data()
    
    if df is not None:
        model, label_encoder, ALL_SYMPTOMS = get_model_and_encoder()
    else:
        model, label_encoder, ALL_SYMPTOMS = None, None, None

# Check if Groq API key is available
try:
    groq_key_available = bool(st.secrets.get("GROQ_API_KEY"))
except:
    groq_key_available = False

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
    
    if groq_key_available:
        st.success("✅ Groq AI Ready")
        st.caption("Providing detailed health information")
    else:
        st.warning("⚠️ Groq AI not configured")
    
    st.write("---")
    st.write("### 📝 Instructions")
    st.write("1. Enter symptoms (comma separated)")
    st.write("2. Click Predict")
    st.write("3. Get detailed health information")

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
                st.caption(f"User symptoms: {', '.join(symptom_list)}")
                
                # Show confidence with color
                if confidence >= 80:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="High")
                elif confidence >= 60:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="Medium")
                else:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="Low")
                
                st.markdown("---")
                
                # Get health information from Groq if available
                if groq_key_available:
                    with st.spinner(f"🤖 Getting detailed health information for {predicted_disease}..."):
                        info = get_health_info_from_groq(predicted_disease, symptom_list, confidence)
                        if info:
                            # Display the formatted information
                            st.markdown(info)
                        else:
                            st.warning("Could not fetch information. Please try again.")
                            
                            # Fallback - show common symptoms from training data
                            disease_data = df[df['prognosis'] == predicted_disease]
                            common_symptoms = []
                            for sym in ALL_SYMPTOMS[:20]:
                                if len(disease_data) > 0 and disease_data[sym].mean() > 0.5:
                                    common_symptoms.append(sym.replace('_', ' ').title())
                            
                            if common_symptoms:
                                st.markdown("**Common symptoms (from training data):**")
                                for sym in common_symptoms[:10]:
                                    st.markdown(f"- {sym}")
                else:
                    st.info("💡 **Groq AI not available.** Add your GROQ_API_KEY to get detailed health information.")
                    
                    # Show common symptoms from training data
                    disease_data = df[df['prognosis'] == predicted_disease]
                    common_symptoms = []
                    for sym in ALL_SYMPTOMS[:20]:
                        if len(disease_data) > 0 and disease_data[sym].mean() > 0.5:
                            common_symptoms.append(sym.replace('_', ' ').title())
                    
                    if common_symptoms:
                        st.markdown("**Common symptoms (from training data):**")
                        for sym in common_symptoms[:10]:
                            st.markdown(f"- {sym}")
                
                st.markdown("---")
                st.caption("⚠️ **Educational purpose only.** Always consult a healthcare provider.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("AI Model: Random Forest | Powered by Machine Learning")
