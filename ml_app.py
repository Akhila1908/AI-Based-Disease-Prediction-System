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
# Extract symptoms from Training.csv dynamically
# ============================================
@st.cache_data
def get_symptom_list():
    """Extract symptom names directly from training data"""
    csv_path = BASE_DIR / 'Training.csv'
    if not csv_path.exists():
        st.error("❌ Training.csv not found! Please upload the dataset.")
        return None
    
    df = pd.read_csv(csv_path)
    # Drop the target column 'prognosis' and any unnamed columns
    symptom_columns = [col for col in df.columns if col not in ['prognosis', 'Unnamed: 133']]
    return symptom_columns

@st.cache_data
def get_disease_info_from_data():
    """Extract disease descriptions from training data if available"""
    csv_path = BASE_DIR / 'Training.csv'
    if not csv_path.exists():
        return {}
    
    df = pd.read_csv(csv_path)
    diseases = df['prognosis'].unique()
    # Return basic info for each disease
    return {disease: {"severity": "Unknown", "remedies": "Consult a healthcare provider"} 
            for disease in diseases}

# Get symptoms from the actual data
ALL_SYMPTOMS = get_symptom_list()

if ALL_SYMPTOMS is None:
    st.stop()

# Get disease info
DISEASE_INFO = get_disease_info_from_data()

# Display count for debugging
st.sidebar.write(f"📊 Loaded {len(ALL_SYMPTOMS)} symptoms from training data")

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
    
    # Train new model
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
    """Convert user input to feature vector using actual symptom list"""
    if not user_input.strip():
        return [0] * len(ALL_SYMPTOMS)
    
    # Clean user input
    user_symptoms = [s.strip().lower().replace(' ', '_') for s in user_input.split(",")]
    
    # Create feature vector
    result = []
    for symptom in ALL_SYMPTOMS:
        # Clean symptom name for comparison (remove extra spaces)
        clean_symptom = symptom.strip().lower().replace('  ', ' ')
        matched = False
        
        for us in user_symptoms:
            # Try exact match and normalized match
            if us == clean_symptom or us.replace('_', ' ') == clean_symptom.replace('_', ' '):
                matched = True
                break
        
        result.append(1 if matched else 0)
    
    # Debug info
    st.sidebar.write(f"📝 Feature vector length: {len(result)}")
    st.sidebar.write(f"✅ Non-zero features: {sum(result)}")
    
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
🟡 **{info.get('severity', 'Consult Doctor')}**

**4. General Home Care Suggestions:**  
• {info.get('remedies', 'Consult a healthcare provider for proper diagnosis')}

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

**Example symptoms you can try:**  
itching, skin_rash, fatigue, headache, nausea, vomiting

**Note:** Use symptom names as they appear in the dataset.
""")

if model is None or label_encoder is None:
    st.error("❌ Model not available. Please check that Training.csv is in the repository.")
    st.stop()

# Example symptoms button
if st.sidebar.button("📋 Load Example Symptoms"):
    st.session_state['symptoms_input'] = "itching, skin_rash, fatigue"
    st.rerun()

# Get symptoms input (use session state for persistence)
symptoms = st.text_area(
    "Enter your symptoms (comma separated):",
    value=st.session_state.get('symptoms_input', ''),
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
                
                # Validate feature count
                expected_features = model.n_features_in_
                if len(input_vector) != expected_features:
                    st.error(f"Feature mismatch: Got {len(input_vector)} features, expected {expected_features}")
                    st.stop()
                
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
