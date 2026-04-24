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

# Import LLM functions
try:
    from llm_helper import get_complete_health_advice, get_symptom_analysis
    LLM_AVAILABLE = True
except Exception as e:
    LLM_AVAILABLE = False
    st.warning(f"LLM features not available: {str(e)}")

# ============================================
# Load data from CSV
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
    
    # Try to load existing model
    if model_path.exists() and encoder_path.exists():
        try:
            model = joblib.load(model_path)
            le = joblib.load(encoder_path)
            return model, le
        except Exception as e:
            st.warning(f"Could not load existing model: {e}")
    
    # Train new model
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
            
            # Save models
            joblib.dump(pipeline, model_path)
            joblib.dump(le, encoder_path)
            
            st.success("✅ Model trained successfully!")
            return pipeline, le
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return None, None

def preprocess_symptoms(user_input, all_symptoms):
    """Convert user input to feature vector"""
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

# ============================================
# UI
# ============================================
st.title("🩺 AI Disease Prediction System")
st.markdown("*Powered by Machine Learning + Groq AI*")

# Sidebar
with st.sidebar:
    st.write("### 📊 Database Statistics")
    st.write(f"**Diseases:** {len(ALL_DISEASES)}")
    st.write(f"**Symptoms:** {len(ALL_SYMPTOMS)}")
    st.write(f"**Training Records:** {len(df)}")
    
    st.write("---")
    st.write("### 🤖 AI Features")
    if LLM_AVAILABLE:
        try:
            from llm_helper import get_groq_client
            client, error = get_groq_client()
            if error:
                st.warning(f"⚠️ Groq AI: {error[:50]}...")
            else:
                st.success("✅ Groq AI is ACTIVE")
                st.caption("Model: Llama 3.3 70B")
                st.caption("Temperature: 0.0")
        except:
            st.warning("⚠️ Groq AI needs configuration")
    else:
        st.warning("⚠️ Groq AI not available")
    
    st.write("---")
    st.write("### 📝 How to Use")
    st.write("1. Enter symptoms (comma separated)")
    st.write("2. Click Predict Disease")
    st.write("3. Get AI-powered health info")
    
    st.write("---")
    if st.button("📋 Load Example"):
        st.session_state['symptoms_input'] = "itching, skin_rash"
        st.rerun()

# Main input
symptoms_input = st.text_area(
    "**Enter your symptoms (comma separated):**",
    value=st.session_state.get('symptoms_input', ''),
    placeholder="Example: itching, skin_rash, fatigue, headache",
    height=100
)

# Predict button
if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
    if not symptoms_input.strip():
        st.warning("⚠️ Please enter at least one symptom")
    elif model is None:
        st.error("❌ Model not available")
    else:
        with st.spinner("🧠 Analyzing symptoms..."):
            try:
                # Preprocess and predict
                input_vector = preprocess_symptoms(symptoms_input, ALL_SYMPTOMS)
                pred_encoded = model.predict([input_vector])[0]
                predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]
                
                # Get confidence
                probs = model.predict_proba([input_vector])[0]
                confidence = max(probs) * 100
                
                # Display prediction
                st.success(f"### 🎯 Predicted Disease: {predicted_disease}")
                st.metric("Confidence Score", f"{confidence:.1f}%")
                
                # Show reported symptoms
                symptom_list = [s.strip() for s in symptoms_input.split(",") if s.strip()]
                st.write(f"**Reported Symptoms:** {', '.join(symptom_list)}")
                
                st.markdown("---")
                
                # Get AI analysis if available
                if LLM_AVAILABLE:
                    with st.spinner("🤖 Generating health information..."):
                        # Get symptom analysis
                        try:
                            analysis = get_symptom_analysis(symptom_list, predicted_disease, confidence)
                            if analysis:
                                st.info(f"**📋 Analysis:** {analysis}")
                        except Exception as e:
                            st.warning(f"Could not generate analysis: {str(e)}")
                        
                        # Get complete health advice
                        try:
                            advice = get_complete_health_advice(predicted_disease)
                            if advice:
                                st.markdown(advice)
                        except Exception as e:
                            st.warning(f"Could not generate health advice: {str(e)}")
                else:
                    st.info("💡 **Note:** Configure Groq AI for detailed health recommendations.")
                    
                    # Show common symptoms from dataset
                    disease_data = df[df['prognosis'] == predicted_disease]
                    common_symptoms = []
                    for sym in ALL_SYMPTOMS[:30]:
                        if len(disease_data) > 0 and disease_data[sym].mean() > 0.5:
                            common_symptoms.append(sym.replace('_', ' ').title())
                    
                    if common_symptoms:
                        st.write("**Common symptoms for this condition:**")
                        cols = st.columns(3)
                        for i, sym in enumerate(common_symptoms[:9]):
                            with cols[i % 3]:
                                st.write(f"- {sym}")
                
                # Disclaimer
                st.markdown("---")
                st.warning("""
                ⚠️ **Medical Disclaimer:** 
                - This is an **educational tool only**
                - Not a substitute for professional medical advice
                - Always consult a qualified healthcare provider
                - In emergencies, contact local emergency services
                """)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🩺 <strong>AI Disease Prediction System</strong></p>
        <p style='font-size: 12px;'>Model: Random Forest | AI: Groq Llama 3.3 70B</p>
        <p style='font-size: 12px;'>⚠️ Educational purpose only</p>
    </div>
    """,
    unsafe_allow_html=True
)
