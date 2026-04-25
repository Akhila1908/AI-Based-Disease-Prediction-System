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
import random
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ============================================
# Free Image Search using Unsplash API (No API Key Required)
# ============================================

def fetch_disease_images_free(disease_name):
    """
    Fetch disease images using Unsplash Source API (completely free, no API key)
    """
    images = []
    
    # Clean the disease name for URL
    search_term = disease_name.lower().replace(' ', '-')
    
    # Unsplash provides random images based on search terms
    # These are high-quality, free stock photos
    
    # Generate multiple unique image URLs
    for i in range(4):
        # Unsplash source URL that returns random images matching the search term
        img_url = f"https://source.unsplash.com/400x300/?{search_term},medical,health"
        images.append(img_url)
    
    return images

# Alternative: Using Pixabay API (free with API key)
def get_pixabay_api_key():
    """Get Pixabay API key from secrets"""
    try:
        return st.secrets.get("PIXABAY_API_KEY")
    except:
        return None

def fetch_disease_images_pixabay(disease_name):
    """
    Fetch disease images using Pixabay API (free up to 100 requests/hour)
    """
    api_key = get_pixabay_api_key()
    
    if not api_key:
        return fetch_disease_images_free(disease_name)
    
    url = "https://pixabay.com/api/"
    
    params = {
        "key": api_key,
        "q": f"{disease_name} medical health",
        "image_type": "photo",
        "per_page": 4,
        "safesearch": "true"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            hits = data.get("hits", [])
            
            images = []
            for hit in hits[:4]:
                # Get medium sized image URL
                img_url = hit.get("webformatURL") or hit.get("largeImageURL")
                if img_url:
                    images.append(img_url)
            
            if images:
                return images
    
    except Exception as e:
        pass
    
    # Fallback to Unsplash
    return fetch_disease_images_free(disease_name)

# Simple fallback - Medical stock photos that are guaranteed to work
def get_fallback_images(disease_name):
    """
    Fallback images that always work (free medical stock photos)
    """
    # Generic medical images that are relevant
    images = [
        "https://cdn.pixabay.com/photo/2020/04/11/12/42/corona-5030706_640.jpg",
        "https://cdn.pixabay.com/photo/2016/11/14/03/14/medical-1822635_640.jpg",
        "https://cdn.pixabay.com/photo/2018/11/15/20/50/doctor-3818689_640.jpg",
        "https://cdn.pixabay.com/photo/2015/10/30/15/56/acne-1015251_640.jpg",
    ]
    return images

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
# Groq LLM Functions
# ============================================

def get_groq_api_key():
    try:
        return st.secrets.get("GROQ_API_KEY")
    except:
        return None

def get_disease_overview(disease_name, symptoms_list, confidence):
    """Get disease overview"""
    
    api_key = get_groq_api_key()
    if not api_key:
        return None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Provide detailed information about {disease_name}.

User symptoms: {', '.join(symptoms_list)}

Provide EXACTLY this format:

ABOUT THE CONDITION:
[2-3 sentences explaining what {disease_name} is]

COMMON SYMPTOMS:
- [symptom 1]
- [symptom 2]
- [symptom 3]
- [symptom 4]
- [symptom 5]

DO THE GIVEN SYMPTOMS MATCH?
[Yes/No] - [brief explanation]

IS IT MILD OR SERIOUS?
[1 sentence explaining severity]

WHEN TO CONSULT A DOCTOR:
- [warning sign 1]
- [warning sign 2]
- [warning sign 3]

Keep responses clear and easy to read."""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": f"You are a medical information assistant. Provide clear, organized health information about {disease_name}. Use the exact format requested. Never give medical advice."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 600
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        return None

def get_home_remedies(disease_name):
    """Get home remedies"""
    
    api_key = get_groq_api_key()
    if not api_key:
        return None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""List 5 home remedies for {disease_name}.

Provide EXACTLY this format:

🌿 HOME REMEDIES:
• [remedy 1]
• [remedy 2]
• [remedy 3]
• [remedy 4]
• [remedy 5]

Keep each remedy short and practical."""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": f"List 5 simple home remedies for {disease_name}. Use bullet points with •. Each on new line."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        return None

def get_diet_recommendations(disease_name):
    """Get diet recommendations"""
    
    api_key = get_groq_api_key()
    if not api_key:
        return None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""List 5 diet recommendations for someone with {disease_name}.

Provide EXACTLY this format:

🥗 DIET RECOMMENDATIONS:
• [recommendation 1]
• [recommendation 2]
• [recommendation 3]
• [recommendation 4]
• [recommendation 5]

Keep each recommendation practical and specific."""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": f"List 5 diet recommendations for {disease_name}. Use bullet points with •. Each on new line."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        return None

def get_prevention_tips(disease_name):
    """Get prevention tips"""
    
    api_key = get_groq_api_key()
    if not api_key:
        return None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""List 5 prevention tips to avoid {disease_name}.

Provide EXACTLY this format:

🛡️ PREVENTION TIPS:
• [tip 1]
• [tip 2]
• [tip 3]
• [tip 4]
• [tip 5]

Keep each tip practical and specific."""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": f"List 5 prevention tips for {disease_name}. Use bullet points with •. Each on new line."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        return None

def get_exercise_guidelines(disease_name):
    """Get exercise guidelines"""
    
    api_key = get_groq_api_key()
    if not api_key:
        return None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Provide exercise guidelines for someone with {disease_name}.

Provide EXACTLY this format:

🏃‍♂️ EXERCISE GUIDELINES:
[2-3 sentences about safe exercises]

Keep it practical and specific."""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": f"Provide exercise guidelines for {disease_name}. Keep it practical."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
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
groq_key_available = get_groq_api_key() is not None

# ============================================
# UI - Simplified Sidebar
# ============================================
st.title("🩺 AI Disease Prediction System")

with st.sidebar:
    st.markdown("### 📝 How to Use")
    st.markdown("1. Enter your symptoms below")
    st.markdown("2. Click **Predict Disease**")
    st.markdown("3. View the predicted disease")
    st.markdown("4. Click on sections to see:")
    st.markdown("   - 🌿 Home Remedies")
    st.markdown("   - 🥗 Diet Recommendations")
    st.markdown("   - 🛡️ Prevention Tips")
    st.markdown("   - 🏃‍♂️ Exercise Guidelines")
    st.markdown("---")
    st.markdown("### 📋 Example Symptoms")
    st.markdown("`itching, skin_rash, fatigue`")
    st.markdown("`cough, fever, runny_nose`")
    st.markdown("`headache, nausea, dizziness`")
    st.markdown("---")
    st.success("✅ Image search ready (Free)")
    st.caption("⚠️ Educational purpose only")

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
                st.caption(f"**Your symptoms:** {', '.join(symptom_list)}")
                
                # Show confidence
                if confidence >= 80:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="High")
                elif confidence >= 60:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="Medium")
                else:
                    st.metric("Confidence", f"{confidence:.0f}%", delta="Low")
                
                # Fetch and display disease images using free method
                with st.spinner(f"📸 Loading images for {predicted_disease}..."):
                    # Try Pixabay if key exists, otherwise use Unsplash
                    pixabay_key = None
                    try:
                        pixabay_key = st.secrets.get("PIXABAY_API_KEY")
                    except:
                        pass
                    
                    if pixabay_key:
                        disease_images = fetch_disease_images_pixabay(predicted_disease)
                    else:
                        disease_images = fetch_disease_images_free(predicted_disease)
                    
                    if disease_images:
                        st.markdown("### 📸 Disease Images")
                        st.markdown("*Medical images for reference*")
                        
                        # Display images in a grid
                        cols = st.columns(2)
                        for idx, img_url in enumerate(disease_images[:4]):
                            with cols[idx % 2]:
                                try:
                                    st.image(img_url, use_container_width=True)
                                except Exception as img_error:
                                    st.write(f"📷 Image {idx + 1} (reference)")
                    else:
                        st.info("📸 Medical images will appear here")
                
                st.markdown("---")
                
                if groq_key_available:
                    # SECTION 1: Disease Overview (always shown)
                    with st.spinner(f"Loading information about {predicted_disease}..."):
                        overview = get_disease_overview(predicted_disease, symptom_list, confidence)
                        if overview:
                            lines = overview.split('\n')
                            
                            for line in lines:
                                line = line.strip()
                                if line:
                                    if line.startswith('ABOUT THE CONDITION:'):
                                        st.markdown("### 📖 About This Condition")
                                    elif line.startswith('COMMON SYMPTOMS:'):
                                        st.markdown("### 🔍 Common Symptoms")
                                    elif line.startswith('DO THE GIVEN SYMPTOMS MATCH?'):
                                        st.markdown("### ✅ Symptom Match")
                                    elif line.startswith('IS IT MILD OR SERIOUS?'):
                                        st.markdown("### 📊 Severity")
                                    elif line.startswith('WHEN TO CONSULT A DOCTOR:'):
                                        st.markdown("### 🏥 When to See a Doctor")
                                    elif line.startswith('-') or line.startswith('•'):
                                        st.markdown(line)
                                    elif line and not line.startswith('```'):
                                        if not line.startswith('ABOUT') and not line.startswith('COMMON') and not line.startswith('DO') and not line.startswith('IS') and not line.startswith('WHEN'):
                                            st.markdown(line)
                            st.markdown("---")
                        else:
                            st.warning("Could not load disease overview")
                    
                    # Expandable sections
                    with st.expander("🌿 Home Remedies", expanded=False):
                        with st.spinner("Loading home remedies..."):
                            remedies = get_home_remedies(predicted_disease)
                            if remedies:
                                for line in remedies.split('\n'):
                                    line = line.strip()
                                    if line:
                                        st.markdown(line)
                            else:
                                st.info("Home remedies information not available")
                    
                    with st.expander("🥗 Diet Recommendations", expanded=False):
                        with st.spinner("Loading diet recommendations..."):
                            diet = get_diet_recommendations(predicted_disease)
                            if diet:
                                for line in diet.split('\n'):
                                    line = line.strip()
                                    if line:
                                        st.markdown(line)
                            else:
                                st.info("Diet recommendations not available")
                    
                    with st.expander("🛡️ Prevention Tips", expanded=False):
                        with st.spinner("Loading prevention tips..."):
                            prevention = get_prevention_tips(predicted_disease)
                            if prevention:
                                for line in prevention.split('\n'):
                                    line = line.strip()
                                    if line:
                                        st.markdown(line)
                            else:
                                st.info("Prevention tips not available")
                    
                    with st.expander("🏃‍♂️ Exercise Guidelines", expanded=False):
                        with st.spinner("Loading exercise guidelines..."):
                            exercise = get_exercise_guidelines(predicted_disease)
                            if exercise:
                                for line in exercise.split('\n'):
                                    line = line.strip()
                                    if line:
                                        st.markdown(line)
                            else:
                                st.info("Exercise guidelines not available")
                
                else:
                    st.info("💡 **Groq AI not available.** Add your GROQ_API_KEY to get detailed health information.")
                    
                    # Show common symptoms from training data
                    disease_data = df[df['prognosis'] == predicted_disease]
                    common_symptoms = []
                    for sym in ALL_SYMPTOMS[:20]:
                        if len(disease_data) > 0 and disease_data[sym].mean() > 0.5:
                            common_symptoms.append(sym.replace('_', ' ').title())
                    
                    if common_symptoms:
                        st.markdown("### 📋 Common Symptoms (from training data)")
                        for sym in common_symptoms[:10]:
                            st.markdown(f"- {sym}")
                
                st.markdown("---")
                st.caption("⚠️ **Educational purpose only.** Always consult a healthcare provider.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("AI Model: Random Forest | Powered by Machine Learning")
