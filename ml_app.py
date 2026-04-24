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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ============================================
# Load everything dynamically from CSV
# ============================================

@st.cache_data
def load_training_data():
    """Load training data from CSV"""
    csv_path = BASE_DIR / 'Training.csv'
    if not csv_path.exists():
        st.error("❌ Training.csv not found!")
        return None
    
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 133'], errors='ignore')
    return df

@st.cache_data
def get_symptom_list(df):
    """Extract symptom names from training data"""
    symptom_columns = [col for col in df.columns if col != 'prognosis']
    return symptom_columns

@st.cache_data
def get_disease_list(df):
    """Extract all unique diseases from training data"""
    return sorted(df['prognosis'].unique())

@st.cache_resource
def load_or_train_model(df):
    """Load existing model or train new one from CSV"""
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
    
    # Train new model from CSV
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
                ('classifier', RandomForestClassifier(
                    n_estimators=200, 
                    random_state=42, 
                    n_jobs=-1
                ))
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
    if not user_input.strip():
        return [0] * len(all_symptoms)
    
    user_symptoms = [s.strip().lower().replace(' ', '_') for s in user_input.split(",")]
    
    result = []
    for symptom in all_symptoms:
        clean_symptom = symptom.strip().lower().replace('  ', ' ')
        matched = any(
            us == clean_symptom or us.replace('_', ' ') == clean_symptom.replace('_', ' ')
            for us in user_symptoms
        )
        result.append(1 if matched else 0)
    
    return result

def get_disease_info_from_csv(df, disease_name):
    """Get sample information about disease from CSV data"""
    # Get rows with this disease
    disease_rows = df[df['prognosis'] == disease_name]
    
    # Find common symptoms for this disease
    symptom_cols = [col for col in df.columns if col != 'prognosis']
    common_symptoms = []
    for symptom in symptom_cols:
        if disease_rows[symptom].mean() > 0.5:  # If >50% of cases have this symptom
            common_symptoms.append(symptom.replace('_', ' ').title())
    
    return {
        "severity": "Varies by case",
        "common_symptoms": common_symptoms[:10],  # Top 10 common symptoms
        "sample_count": len(disease_rows)
    }

# ============================================
# Load data and initialize
# ============================================
df = load_training_data()

if df is None:
    st.stop()

ALL_SYMPTOMS = get_symptom_list(df)
ALL_DISEASES = get_disease_list(df)
model, label_encoder = load_or_train_model(df)

# Sidebar stats
st.sidebar.title("📊 Database Info")
st.sidebar.metric("Total Diseases", len(ALL_DISEASES))
st.sidebar.metric("Total Symptoms", len(ALL_SYMPTOMS))
st.sidebar.metric("Training Samples", len(df))
st.sidebar.markdown("---")

# ============================================
# Main UI
# ============================================
st.title("🩺 AI-Based Disease Prediction System")
st.markdown("*Predict diseases based on symptoms using Machine Learning*")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Disease Predictor", "📚 Disease Library", "ℹ️ About"])

# ============================================
# TAB 1: Disease Predictor
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Symptoms")
        
        symptoms_input = st.text_area(
            "List symptoms separated by commas:",
            placeholder="Example: itching, skin_rash, fatigue, headache",
            height=100
        )
        
        # Quick select common symptoms
        st.markdown("**Quick Add Common Symptoms:**")
        common_symptoms = ALL_SYMPTOMS[:20]  # First 20 symptoms
        cols = st.columns(5)
        for i, symptom in enumerate(common_symptoms[:10]):
            with cols[i % 5]:
                if st.button(f"➕ {symptom.replace('_', ' ').title()}", key=f"btn_{symptom}"):
                    if symptoms_input:
                        symptoms_input += f", {symptom}"
                    else:
                        symptoms_input = symptom
                    st.rerun()
    
    with col2:
        st.subheader("How to Use")
        st.info("""
        1. Enter your symptoms
        2. Click 'Predict Disease'
        3. AI analyzes the pattern
        4. Get prediction results
        """)
        
        if symptoms_input:
            symptom_count = len([s for s in symptoms_input.split(",") if s.strip()])
            st.metric("Symptoms Entered", symptom_count)
    
    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        if not symptoms_input.strip():
            st.warning("⚠️ Please enter at least one symptom")
        elif model is None:
            st.error("❌ Model not available")
        else:
            with st.spinner("🧠 AI Analyzing your symptoms..."):
                try:
                    input_vector = preprocess_symptoms(symptoms_input, ALL_SYMPTOMS)
                    
                    # Validate feature count
                    if len(input_vector) != model.n_features_in_:
                        st.error(f"Feature mismatch. Expected {model.n_features_in_}, got {len(input_vector)}")
                    else:
                        # Get prediction
                        prediction_encoded = model.predict([input_vector])[0]
                        predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]
                        
                        # Get confidence
                        probabilities = model.predict_proba([input_vector])[0]
                        confidence = max(probabilities) * 100
                        
                        # Display result
                        st.success(f"### 🎯 Predicted Disease: {predicted_disease}")
                        st.metric("Confidence Score", f"{confidence:.1f}%")
                        
                        # Get disease info from CSV
                        disease_info = get_disease_info_from_csv(df, predicted_disease)
                        
                        # Show common symptoms for this disease
                        if disease_info['common_symptoms']:
                            st.markdown("### 📋 Common Symptoms for this Condition")
                            cols = st.columns(3)
                            for i, symptom in enumerate(disease_info['common_symptoms']):
                                with cols[i % 3]:
                                    st.markdown(f"- {symptom}")
                        
                        st.markdown("---")
                        st.markdown("### 💡 Next Steps")
                        st.markdown("""
                        - **Monitor** your symptoms for 24-48 hours
                        - **Rest** and stay hydrated
                        - **Consult** a healthcare provider if symptoms persist
                        - **Keep** a symptom diary for your doctor
                        """)
                        
                        # Download report button
                        report = f"""
Disease Prediction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Predicted Disease: {predicted_disease}
Confidence: {confidence:.1f}%
Symptoms Reported: {symptoms_input}

Common Symptoms for {predicted_disease}:
{', '.join(disease_info['common_symptoms'][:10])}

Disclaimer: This is an AI prediction tool. Always consult a healthcare provider.
"""
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

# ============================================
# TAB 2: Disease Library
# ============================================
with tab2:
    st.subheader("📚 Disease Information Library")
    st.markdown(f"*{len(ALL_DISEASES)} diseases available in the database*")
    
    # Search and filter
    search = st.text_input("🔍 Search for a disease:", placeholder="Type disease name...")
    
    if search:
        filtered_diseases = [d for d in ALL_DISEASES if search.lower() in d.lower()]
    else:
        filtered_diseases = ALL_DISEASES
    
    st.markdown(f"**Showing {len(filtered_diseases)} diseases**")
    
    # Display diseases in grid
    cols_per_row = 3
    for i in range(0, len(filtered_diseases), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, disease in enumerate(filtered_diseases[i:i+cols_per_row]):
            with cols[j]:
                with st.expander(f"📖 {disease}"):
                    info = get_disease_info_from_csv(df, disease)
                    st.markdown(f"**📊 Severity:** {info['severity']}")
                    st.markdown(f"**📈 Sample Cases:** {info['sample_count']} in dataset")
                    
                    if info['common_symptoms']:
                        st.markdown("**🔍 Common Symptoms:**")
                        for symptom in info['common_symptoms'][:5]:
                            st.markdown(f"- {symptom}")

# ============================================
# TAB 3: About
# ============================================
with tab3:
    st.subheader("ℹ️ About This System")
    
    st.markdown("""
    ### How It Works
    
    This AI-powered system uses **Machine Learning** to predict potential diseases based on symptoms:
    
    1. **Training Data**: The model is trained on `{len(df)}` patient records with `{len(ALL_SYMPTOMS)}` different symptoms
    2. **Algorithm**: Random Forest Classifier with 200 decision trees
    3. **Accuracy**: The model learns patterns between symptoms and diseases
    
    ### Dataset Statistics
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diseases", len(ALL_DISEASES))
    with col2:
        st.metric("Symptoms", len(ALL_SYMPTOMS))
    with col3:
        st.metric("Training Records", len(df))
    
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Important Disclaimer
    
    **This tool is for EDUCATIONAL PURPOSES only.**
    
    - Not a substitute for professional medical advice
    - AI predictions may not be 100% accurate
    - Always consult a qualified healthcare provider
    - In emergencies, contact local emergency services
    
    ### 📊 Model Performance
    
    - **Algorithm**: Random Forest Classifier
    - **Training Size**: {len(df)} samples
    - **Features**: {len(ALL_SYMPTOMS)} binary symptoms
    - **Validation**: Stratified train-test split
    """.format(len(df)=len(df), len(ALL_SYMPTOMS)=len(ALL_SYMPTOMS)))
    
    st.markdown("---")
    st.caption(f"Built with Streamlit & Scikit-learn | Data Source: Training.csv | Last Updated: {datetime.now().strftime('%Y-%m-%d')}")

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🩺 <strong>AI Disease Prediction System</strong> | Powered by Machine Learning</p>
        <p style='font-size: 12px;'>⚠️ This is an educational tool. Always consult a healthcare provider.</p>
    </div>
    """,
    unsafe_allow_html=True
)
