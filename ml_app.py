# ============================================
# MUST BE FIRST - Streamlit page config
# ============================================
import streamlit as st
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="centered"  # Changed from "wide" to "centered" for smaller width
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
# Groq LLM Setup - FIXED
# ============================================

def get_groq_client():
    """Initialize Groq client - FIXED VERSION"""
    try:
        # Try multiple ways to get API key
        api_key = None
        
        # Method 1: Streamlit secrets
        try:
            if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
                print("✅ Found API key in Streamlit secrets")
        except:
            pass
        
        # Method 2: Environment variable
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                print("✅ Found API key in environment variables")
        
        if not api_key:
            return None, "GROQ_API_KEY not found. Please add to Streamlit secrets."
        
        # Initialize Groq client
        from groq import Groq
        client = Groq(api_key=api_key)
        return client, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_disease_info_from_groq(disease_name, symptoms_list, confidence):
    """Get disease-specific information from Groq LLM"""
    
    client, error = get_groq_client()
    
    if error or client is None:
        return None
    
    prompt = f"""Provide health information for {disease_name} based on these symptoms: {', '.join(symptoms_list)}.

Provide ONLY in this exact format:

🌿 HOME REMEDIES:
• [specific remedy 1 for {disease_name}]
• [specific remedy 2 for {disease_name}]
• [specific remedy 3 for {disease_name}]
• [specific remedy 4 for {disease_name}]
• [specific remedy 5 for {disease_name}]

🥗 DIET RECOMMENDATIONS:
• [diet tip 1 for {disease_name}]
• [diet tip 2 for {disease_name}]
• [diet tip 3 for {disease_name}]
• [diet tip 4 for {disease_name}]
• [diet tip 5 for {disease_name}]

🛡️ PREVENTION TIPS:
• [prevention tip 1 for {disease_name}]
• [prevention tip 2 for {disease_name}]
• [prevention tip 3 for {disease_name}]
• [prevention tip 4 for {disease_name}]
• [prevention tip 5 for {disease_name}]

🏃‍♂️ EXERCISE GUIDELINES:
[2 sentences about safe exercises for {disease_name}]

📚 WHEN TO SEE A DOCTOR:
[2-3 warning signs specific to {disease_name}]

Keep responses practical and specific to {disease_name}."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a medical information assistant. Provide specific, practical information about {disease_name} only. Never give medical advice."
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
# Load data from CSV
# ============================================
@st.cache_data
def load_training_data():
    csv_path = BASE_DIR / 'Training.csv'
    if not csv_path.exists():
        st.error("❌ Training.csv not found!")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        # Remove unnamed column if exists
        df = df.drop(columns=['Unnamed: 133'], errors='ignore')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

@st.cache_data
def get_symptom_list(df):
    if df is None:
        return []
    return [col for col in df.columns if col != 'prognosis']

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
            st.warning(f"Training new model...")
    
    # Train new model
    with st.spinner("🔄 Training AI model..."):
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
            
            return pipeline, le
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return None, None

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
# Load data
# ============================================
df = load_training_data()

if df is None:
    st.stop()

ALL_SYMPTOMS = get_symptom_list(df)
model, label_encoder = load_or_train_model(df)

# ============================================
# UI
# ============================================
st.title("🩺 AI Disease Prediction System")

# Sidebar
with st.sidebar:
    st.write("### 📊 Dataset Info")
    if df is not None:
        st.write(f"**Diseases:** {df['prognosis'].nunique()}")
        st.write(f"**Symptoms:** {len(ALL_SYMPTOMS)}")
        st.write(f"**Records:** {len(df)}")
    
    st.write("---")
    
    # Test Groq connection
    client, error = get_groq_client()
    if client:
        st.success("✅ Groq AI Ready")
    else:
        st.warning("⚠️ Groq AI: " + (error[:50] if error else "Not configured"))
    
    st.write("---")
    st.write("### 📝 How to Use")
    st.write("1. Enter symptoms (comma separated)")
    st.write("2. Click Predict")
    st.write("3. Get AI-powered health info")

# Main input
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
                
                # Show confidence with color coding
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
                    with st.spinner(f"🤖 Getting information about {predicted_disease}..."):
                        disease_info = get_disease_info_from_groq(predicted_disease, symptom_list, confidence)
                        
                        if disease_info:
                            st.markdown(disease_info)
                        else:
                            st.error(f"Could not fetch information. Please try again.")
                else:
                    st.info("💡 **Enable Groq AI:** Add GROQ_API_KEY to Streamlit secrets for home remedies and health tips.")
                    
                    # Show common symptoms from training data
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
                
                # Disclaimer
                st.markdown("---")
                st.caption("⚠️ **Educational purpose only.** Always consult a healthcare provider.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("AI Model: Random Forest | Powered by Machine Learning")
