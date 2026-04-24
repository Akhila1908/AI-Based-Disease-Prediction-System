# ============================================
# MUST BE FIRST - Streamlit page config
# ============================================
import streamlit as st
st.set_page_config(
    page_title="AI Disease Prediction & Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
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
import random
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__.parent)

# ============================================
# COMPLETE DISEASE DATABASE (Home Remedies, Diet, Exercise, Prevention)
# ============================================

DISEASE_COMPLETE_INFO = {
    "Acne": {
        "severity": "Mild to Moderate",
        "home_remedies": [
            "🌿 Apply tea tree oil diluted with coconut oil",
            "🍯 Use honey and cinnamon mask for 15 minutes",
            "🌱 Apply fresh aloe vera gel daily",
            "🧊 Ice cubes to reduce inflammation and redness",
            "🍋 Lemon juice with rose water as toner"
        ],
        "prevention": [
            "Wash face twice daily with gentle cleanser",
            "Avoid touching your face throughout the day",
            "Change pillowcases every 2-3 days",
            "Remove makeup before sleeping",
            "Use non-comedogenic products"
        ],
        "diet": [
            "🥗 Low glycemic index foods (whole grains, legumes)",
            "🥜 Zinc-rich foods (nuts, seeds, pumpkin seeds)",
            "🐟 Omega-3 fatty acids (salmon, walnuts, flaxseeds)",
            "🥕 Vitamin A rich foods (carrots, sweet potatoes)",
            "🥛 Probiotics (yogurt, kefir, kimchi)"
        ],
        "exercise": "🧘 Gentle yoga and light cardio; shower immediately after sweating",
        "awareness": "Acne affects 85% of teens and young adults. It's caused by hormones, not poor hygiene. Stress can worsen acne.",
        "warning_signs": "Severe cysts, scarring, or if acne affects self-esteem"
    },
    "Fungal infection": {
        "severity": "Mild to Moderate",
        "home_remedies": [
            "🥥 Apply virgin coconut oil 2-3 times daily",
            "🍎 Use diluted apple cider vinegar (1:3 ratio)",
            "🌿 Tea tree oil mixed with carrier oil",
            "🧄 Crushed garlic paste (external use only)",
            "🧴 Keep affected area completely dry"
        ],
        "prevention": [
            "Keep skin clean and thoroughly dry",
            "Wear breathable cotton fabrics",
            "Never share towels, clothes, or combs",
            "Change socks daily, use antifungal powder",
            "Wear flip-flops in public showers/pools"
        ],
        "diet": [
            "🍚 Reduce sugar and yeast intake",
            "🥛 Probiotics (yogurt, kefir, kombucha)",
            "🧄 Fresh garlic and onion daily",
            "🥥 Coconut oil in cooking",
            "🍎 Apple cider vinegar drinks (1 tbsp in water)"
        ],
        "exercise": "🏃 Light exercises; shower and dry thoroughly immediately after",
        "awareness": "Fungal infections thrive in warm, moist areas. Athletes and gym-goers are at higher risk.",
        "warning_signs": "Spreading rash, fever, or if over-the-counter treatments don't work"
    },
    "Common Cold": {
        "severity": "Mild",
        "home_remedies": [
            "🍯 Warm honey and ginger tea 3 times daily",
            "💨 Steam inhalation with eucalyptus oil",
            "🧂 Salt water gargle (1/2 tsp salt in warm water)",
            "🍲 Hot chicken soup (proven to help)",
            "🍋 Warm lemon water with honey and turmeric"
        ],
        "prevention": [
            "🧼 Wash hands frequently with soap",
            "🙌 Avoid touching eyes, nose, and mouth",
            "😴 Get 7-8 hours of sleep nightly",
            "💧 Stay hydrated (8-10 glasses water)",
            "🍊 Take vitamin C regularly"
        ],
        "diet": [
            "🍊 Vitamin C rich foods (oranges, kiwi, bell peppers)",
            "🥜 Zinc-rich foods (pumpkin seeds, chickpeas)",
            "🥣 Warm soups and broths",
            "🧄 Fresh garlic has antiviral properties",
            "🫚 Ginger tea for sore throat"
        ],
        "exercise": "🚶 Light walking only; avoid intense exercise during fever",
        "awareness": "Colds are viral - antibiotics don't work. Most resolve in 7-10 days with rest.",
        "warning_signs": "Fever over 101°F for 3+ days, difficulty breathing, chest pain"
    },
    "Migraine": {
        "severity": "Moderate to Severe",
        "home_remedies": [
            "❄️ Cold compress on forehead and temples",
            "🌑 Rest in complete darkness and silence",
            "☕ Small amount of caffeine (tea/coffee) at onset",
            "🌿 Peppermint oil massage on temples",
            "💧 Stay very hydrated, add electrolytes"
        ],
        "prevention": [
            "📓 Keep migraine diary to identify triggers",
            "⏰ Maintain consistent sleep schedule",
            "🧘 Practice stress management (meditation)",
            "🍽️ Don't skip meals, eat regularly",
            "🚫 Avoid known triggers (aged cheese, red wine)"
        ],
        "diet": [
            "🥬 Magnesium-rich foods (spinach, almonds)",
            "🍌 Avoid aged cheeses and processed meats",
            "☕ Limit caffeine to small amounts",
            "💧 Stay hydrated with water and electrolytes",
            "🍽️ Small frequent meals to prevent blood sugar drops"
        ],
        "exercise": "🧘 Gentle yoga, stretching, walking; avoid high-intensity during attack",
        "awareness": "Migraines affect 1 in 7 people worldwide. Women are 3x more likely to get them.",
        "warning_signs": "Sudden severe headache, confusion, vision changes, slurred speech"
    },
    "Diabetes": {
        "severity": "Chronic",
        "home_remedies": [
            "🌿 Fenugreek seeds soaked overnight, eat in morning",
            "☕ Cinnamon tea (1/2 tsp cinnamon in warm water)",
            "🥒 Bitter gourd (karela) juice",
            "🌱 Aloe vera juice on empty stomach",
            "🍃 Indian gooseberry (amla) juice"
        ],
        "prevention": [
            "⚖️ Maintain healthy BMI (18.5-24.9)",
            "🏃‍♂️ 30 minutes exercise 5 days/week",
            "🍽️ Balanced diet with portion control",
            "📊 Regular blood sugar monitoring",
            "🍬 Avoid processed sugars and refined carbs"
        ],
        "diet": [
            "🌾 High fiber foods (oats, whole grains)",
            "🥬 Leafy greens (spinach, kale)",
            "🍗 Lean proteins (chicken, fish, tofu)",
            "🥑 Low glycemic index fruits (berries, apple)",
            "🥜 Nuts and seeds for healthy fats"
        ],
        "exercise": "🚶 Walking 30 mins daily, strength training, yoga, swimming",
        "awareness": "India is diabetes capital of the world. Early detection prevents complications.",
        "warning_signs": "Excessive thirst, frequent urination, blurred vision, slow healing"
    },
    "Hypertension": {
        "severity": "Chronic",
        "home_remedies": [
            "🧄 Eat 2-3 raw garlic cloves daily",
            "🌺 Hibiscus tea (2 cups daily)",
            "🍌 Eat bananas rich in potassium",
            "🍫 Small amount dark chocolate (70%+ cocoa)",
            "💧 Drink coconut water"
        ],
        "prevention": [
            "🧂 Reduce sodium (less than 2300mg daily)",
            "🏃‍♂️ Regular aerobic exercise",
            "🧘 Stress management techniques",
            "🍷 Limit alcohol consumption",
            "🚭 Quit smoking completely"
        ],
        "diet": [
            "🥗 DASH diet (fruits, vegetables, low-fat dairy)",
            "🥬 Leafy greens (spinach, kale)",
            "🫐 Berries rich in anthocyanins",
            "🌰 Beets and beet juice",
            "🥛 Low-fat dairy for calcium"
        ],
        "exercise": "🏊‍♂️ Brisk walking 30 mins, swimming, cycling, yoga",
        "awareness": "Hypertension is the 'silent killer' - no symptoms until damage occurs.",
        "warning_signs": "Severe headache, chest pain, vision problems, difficulty breathing"
    },
    "Tuberculosis": {
        "severity": "Severe",
        "home_remedies": [
            "🧄 Eat 2-3 raw garlic cloves on empty stomach",
            "🫚 Ginger tea with honey 3 times daily",
            "🥛 Turmeric milk (haldi doodh) at bedtime",
            "🥚 High protein nutritious diet",
            "💤 Complete bed rest during initial treatment"
        ],
        "prevention": [
            "💉 BCG vaccination for children",
            "😷 Avoid crowded, poorly ventilated places",
            "🪟 Ensure good home ventilation",
            "🤧 Cover mouth while coughing",
            "💊 Complete full TB treatment course"
        ],
        "diet": [
            "🥚 High protein foods (eggs, paneer, chicken)",
            "🍊 Vitamin C rich foods (amla, citrus)",
            "🥬 Iron-rich foods (spinach, beetroot)",
            "🥛 Calcium-rich foods (milk, curd)",
            "🌾 Whole grains for energy"
        ],
        "exercise": "🛌 Rest completely during active infection. Light walking only after recovery starts.",
        "awareness": "TB is curable with proper treatment. Never miss DOTS medication doses.",
        "warning_signs": "Cough >3 weeks, blood in sputum, night sweats, weight loss"
    },
    "Malaria": {
        "severity": "Severe",
        "home_remedies": [
            "🌿 Cinchona bark tea (natural quinine)",
            "🧄 Garlic and honey mixture",
            "🫚 Ginger tea for fever",
            "🥤 Stay hydrated with ORS solution",
            "🛌 Complete bed rest"
        ],
        "prevention": [
            "🦟 Use mosquito nets while sleeping",
            "🧴 Apply mosquito repellent (DEET/permethrin)",
            "🚫 Eliminate standing water around home",
            " Wear long sleeves at dusk/dawn",
            "🪟 Install window screens"
        ],
        "diet": [
            "🍲 Easy-to-digest soups and broths",
            "🍌 Potassium-rich foods (bananas)",
            "🥚 Soft boiled eggs for protein",
            "🍚 Rice and dal for energy",
            "💧 Oral rehydration solution"
        ],
        "exercise": "🛌 Complete rest during fever. No exercise until fully recovered.",
        "awareness": "Malaria kills over 400,000 people annually. Early treatment saves lives.",
        "warning_signs": "High fever with chills, severe headache, vomiting, confusion"
    },
    "Dengue": {
        "severity": "Severe",
        "home_remedies": [
            "🥣 Papaya leaf juice (increases platelets)",
            "💧 Coconut water for electrolytes",
            "🥛 Barley water for hydration",
            "🍵 Giloy (Tinospora cordifolia) juice",
            "🛌 Complete bed rest"
        ],
        "prevention": [
            "🦟 Use mosquito nets and repellents",
            "🚫 Eliminate stagnant water",
            " Wear full sleeve clothing",
            "🪟 Use mosquito screens on windows",
            "🧴 Apply repellent containing DEET"
        ],
        "diet": [
            "💧 ORS solution for hydration",
            "🥣 Easy-to-digest soups",
            "🍌 Potassium-rich foods (bananas)",
            "🥚 Soft protein (eggs, tofu)",
            "🍎 Apple and pear puree"
        ],
        "exercise": "🛌 Complete bed rest. No exercise for 2-4 weeks after recovery.",
        "awareness": "Dengue can become severe within hours. Watch for warning signs.",
        "warning_signs": "Severe abdominal pain, persistent vomiting, bleeding gums, lethargy"
    },
    "Typhoid": {
        "severity": "Severe",
        "home_remedies": [
            "🍌 Mashed bananas for easy digestion",
            "🍚 Rice gruel (congee) for energy",
            "💧 ORS solution for hydration",
            "🥥 Tender coconut water",
            "🛌 Strict bed rest"
        ],
        "prevention": [
            "💉 Get typhoid vaccination",
            "🧼 Wash hands before eating",
            "💧 Drink only boiled/purified water",
            "🍽️ Avoid street food",
            "🥗 Eat freshly cooked hot food"
        ],
        "diet": [
            "🍚 Soft rice and dal",
            "🥣 Clear soups and broths",
            "🍌 Mashed bananas",
            "🥚 Soft boiled eggs",
            "💧 Oral rehydration solution"
        ],
        "exercise": "🛌 Complete bed rest during fever. Slow recovery after.",
        "awareness": "Typhoid spreads through contaminated food/water. Hand hygiene is key.",
        "warning_signs": "Very high fever (104°F), rose-colored spots, severe stomach pain"
    },
    "Gastroenteritis": {
        "severity": "Moderate",
        "home_remedies": [
            "💧 ORS solution (1 liter water + 6 tsp sugar + 1/2 tsp salt)",
            "🥣 Rice water (congee) for hydration",
            "🫚 Ginger tea for nausea",
            "🌿 Mint leaves in boiling water",
            "🥥 Tender coconut water"
        ],
        "prevention": [
            "🧼 Wash hands before eating and after toilet",
            "💧 Drink only purified water",
            "🍽️ Avoid raw/undercooked food",
            "🧊 Refrigerate food properly",
            "🚫 Don't share utensils"
        ],
        "diet": [
            "🍌 BRAT diet (Bananas, Rice, Applesauce, Toast)",
            "🥣 Plain rice porridge",
            "🍞 Dry toast or crackers",
            "🥔 Boiled potatoes without skin",
            "💧 Clear fluids (water, ORS, coconut water)"
        ],
        "exercise": "🛌 Complete rest. No exercise until diarrhea stops.",
        "awareness": "Gastroenteritis is highly contagious. Isolation prevents spread.",
        "warning_signs": "Blood in stool, high fever, severe dehydration, no urination"
    },
    "Pneumonia": {
        "severity": "Severe",
        "home_remedies": [
            "💨 Steam inhalation with eucalyptus",
            "🍯 Honey and warm water for cough",
            "🫚 Ginger tea for inflammation",
            "🧄 Garlic in warm water",
            "🛏️ Complete bed rest"
        ],
        "prevention": [
            "💉 Get pneumonia vaccine",
            "🧼 Wash hands frequently",
            "🚭 Avoid smoking and secondhand smoke",
            "🥗 Boost immunity with healthy diet",
            "😷 Wear mask in crowded places"
        ],
        "diet": [
            "🥣 Warm soups and broths",
            "🍗 High protein foods (chicken, eggs)",
            "🍊 Vitamin C rich fruits",
            "🥛 Warm milk with turmeric",
            "💧 Stay hydrated"
        ],
        "exercise": "🛌 Complete rest. Breathing exercises only after recovery.",
        "awareness": "Pneumonia is the leading infectious cause of death in children worldwide.",
        "warning_signs": "Difficulty breathing, chest pain, high fever, confusion, blue lips"
    }
}

# Default info for any disease not in database
DEFAULT_INFO = {
    "severity": "Consult Doctor",
    "home_remedies": [
        "🩺 Consult healthcare provider for proper diagnosis",
        "📋 Follow prescribed treatment plan",
        "💊 Take medications as directed",
        "🛌 Get adequate rest",
        "💧 Stay well hydrated"
    ],
    "prevention": [
        "Regular health check-ups",
        "Maintain healthy lifestyle",
        "Balanced diet and exercise",
        "Adequate sleep (7-8 hours)",
        "Stress management"
    ],
    "diet": [
        "🥗 Balanced diet with fruits and vegetables",
        "💧 Stay hydrated (8-10 glasses water)",
        "🥩 Lean proteins for recovery",
        "🌾 Whole grains for energy",
        "🥛 Probiotics for gut health"
    ],
    "exercise": "🚶 Light walking; consult doctor before starting any exercise routine",
    "awareness": "Always consult a healthcare provider for proper diagnosis and treatment.",
    "warning_signs": "Seek immediate medical help if symptoms worsen or new symptoms appear"
}

# ============================================
# Load data from CSV
# ============================================
@st.cache_data
def load_training_data():
    """Load training data from CSV"""
    csv_path = BASE_DIR / 'Training.csv'
    if not csv_path.exists():
        st.warning("⚠️ Training.csv not found! Using demo mode.")
        return None
    
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 133'], errors='ignore')
    return df

@st.cache_data
def get_symptom_list(df):
    """Extract symptom names from training data"""
    if df is None:
        return []
    return [col for col in df.columns if col != 'prognosis']

@st.cache_data
def get_disease_list(df):
    """Extract all unique diseases from training data"""
    if df is None:
        return list(DISEASE_COMPLETE_INFO.keys())
    return sorted(df['prognosis'].unique())

@st.cache_resource
def load_or_train_model(df):
    """Load existing model or train new one"""
    model_path = BASE_DIR / "disease_model.joblib"
    encoder_path = BASE_DIR / "label_encoder.joblib"
    
    # If no CSV, we can't train
    if df is None:
        return None, None
    
    # Try to load existing model
    if model_path.exists() and encoder_path.exists():
        try:
            model = joblib.load(model_path)
            le = joblib.load(encoder_path)
            return model, le
        except Exception as e:
            st.warning(f"Could not load model: {e}")
    
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
                ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
            ])
            
            pipeline.fit(X, y_encoded)
            
            joblib.dump(pipeline, model_path)
            joblib.dump(le, encoder_path)
            
            return pipeline, le
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return None, None

def get_disease_info(disease_name):
    """Get complete disease information"""
    if disease_name in DISEASE_COMPLETE_INFO:
        return DISEASE_COMPLETE_INFO[disease_name]
    return DEFAULT_INFO

def preprocess_symptoms(user_input, all_symptoms):
    """Convert user input to feature vector"""
    if not user_input.strip() or not all_symptoms:
        return [0] * 132
    
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

# ============================================
# Load data
# ============================================
df = load_training_data()
ALL_SYMPTOMS = get_symptom_list(df)
ALL_DISEASES = get_disease_list(df)
model, label_encoder = load_or_train_model(df)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("🏥 Health Assistant")
    st.markdown("---")
    
    if df is not None:
        st.metric("📊 Diseases", len(ALL_DISEASES))
        st.metric("📋 Symptoms", len(ALL_SYMPTOMS))
        st.metric("📈 Training Data", len(df))
    else:
        st.info("📚 Demo Mode - 41 Diseases Available")
    
    st.markdown("---")
    st.markdown("### 💡 Daily Health Tip")
    tips = [
        "Drink 8 glasses of water daily",
        "Get 7-8 hours of sleep",
        "Exercise for 30 minutes daily",
        "Eat 5 servings of fruits/vegetables",
        "Take short breaks from screens"
    ]
    st.info(random.choice(tips))
    
    st.markdown("---")
    st.markdown("### 📞 Emergency")
    st.markdown("**Ambulance:** 108/102")
    st.markdown("**Police:** 100")
    st.markdown("**National Helpline:** 112")

# ============================================
# Main UI with Tabs
# ============================================
st.title("🩺 AI Health Assistant")
st.markdown("*Your complete health companion for symptom analysis and wellness guidance*")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Disease Predictor",
    "💊 Disease Library",
    "🌿 Home Remedies",
    "🥗 Diet & Exercise",
    "📚 Health Tips"
])

# ============================================
# TAB 1: Disease Predictor
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Your Symptoms")
        
        symptoms_input = st.text_area(
            "List symptoms separated by commas:",
            placeholder="Example: itching, skin_rash, fatigue, headache, fever",
            height=100
        )
        
        if ALL_SYMPTOMS:
            st.markdown("**Quick Add Common Symptoms:**")
            common = ALL_SYMPTOMS[:15]
            cols = st.columns(5)
            for i, sym in enumerate(common):
                with cols[i % 5]:
                    if st.button(f"➕ {sym.replace('_', ' ').title()}", key=f"q{sym}"):
                        if symptoms_input:
                            symptoms_input += f", {sym}"
                        else:
                            symptoms_input = sym
                        st.rerun()
    
    with col2:
        st.subheader("ℹ️ How It Works")
        st.info("""
        1️⃣ Enter your symptoms  
        2️⃣ AI analyzes patterns  
        3️⃣ Get disease prediction  
        4️⃣ View remedies & tips  
        """)
        
        if symptoms_input:
            count = len([s for s in symptoms_input.split(",") if s.strip()])
            st.metric("Symptoms Entered", count)
    
    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        if not symptoms_input.strip():
            st.warning("⚠️ Please enter at least one symptom")
        elif model is None and df is None:
            st.info("📚 Demo Mode - Showing sample prediction")
            
            # Demo prediction
            demo_disease = random.choice(ALL_DISEASES)
            st.success(f"### 🎯 Predicted Disease: {demo_disease}")
            
            info = get_disease_info(demo_disease)
            
            with st.expander("🌿 Home Remedies", expanded=True):
                for remedy in info["home_remedies"]:
                    st.markdown(remedy)
            
            with st.expander("🛡️ Prevention Tips"):
                for prev in info["prevention"]:
                    st.markdown(f"• {prev}")
            
            with st.expander("🥗 Recommended Diet"):
                for diet in info["diet"]:
                    st.markdown(f"• {diet}")
            
            st.info(f"**Exercise:** {info['exercise']}")
            st.warning(f"**Warning Signs:** {info['warning_signs']}")
            
        elif model and label_encoder:
            with st.spinner("🧠 AI Analyzing..."):
                try:
                    input_vector = preprocess_symptoms(symptoms_input, ALL_SYMPTOMS)
                    
                    if len(input_vector) != model.n_features_in_:
                        st.error(f"Feature mismatch: Expected {model.n_features_in_}, got {len(input_vector)}")
                    else:
                        pred_encoded = model.predict([input_vector])[0]
                        predicted = label_encoder.inverse_transform([pred_encoded])[0]
                        
                        probs = model.predict_proba([input_vector])[0]
                        confidence = max(probs) * 100
                        
                        st.success(f"### 🎯 Predicted: {predicted}")
                        st.metric("Confidence", f"{confidence:.1f}%")
                        
                        info = get_disease_info(predicted)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("### 🌿 Home Remedies")
                            for remedy in info["home_remedies"]:
                                st.markdown(remedy)
                            
                            st.markdown("### 🛡️ Prevention")
                            for prev in info["prevention"]:
                                st.markdown(f"• {prev}")
                        
                        with col_b:
                            st.markdown("### 🥗 Diet Recommendations")
                            for diet in info["diet"]:
                                st.markdown(f"• {diet}")
                            
                            st.markdown("### 🏃‍♂️ Exercise")
                            st.info(info["exercise"])
                        
                        st.markdown("---")
                        st.markdown("### 📢 Awareness")
                        st.info(info["awareness"])
                        
                        st.warning(f"**⚠️ Warning Signs:** {info['warning_signs']}")
                        
                        # Download report
                        report = f"""
Disease Prediction Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Predicted Disease: {predicted}
Confidence: {confidence:.1f}%
Symptoms: {symptoms_input}

Home Remedies:
{chr(10).join(info['home_remedies'])}

Prevention Tips:
{chr(10).join(['• ' + p for p in info['prevention']])}

Diet Recommendations:
{chr(10).join(['• ' + d for d in info['diet']])}

Exercise: {info['exercise']}

Awareness: {info['awareness']}

Warning Signs: {info['warning_signs']}

Disclaimer: This is for educational purposes only. Consult a doctor.
"""
                        st.download_button("📥 Download Report", report, f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================
# TAB 2: Disease Library
# ============================================
with tab2:
    st.subheader("📚 Complete Disease Library")
    
    search = st.text_input("🔍 Search Disease:", placeholder="Type disease name...")
    
    if search:
        filtered = [d for d in ALL_DISEASES if search.lower() in d.lower()]
    else:
        filtered = ALL_DISEASES
    
    st.markdown(f"**{len(filtered)} diseases found**")
    
    cols = st.columns(3)
    for i, disease in enumerate(filtered):
        with cols[i % 3]:
            with st.expander(f"📖 {disease}"):
                info = get_disease_info(disease)
                st.markdown(f"**Severity:** {info['severity']}")
                st.markdown("**Quick Remedies:**")
                for r in info["home_remedies"][:2]:
                    st.markdown(f"- {r}")
                if st.button(f"View Full Info", key=f"view_{disease}"):
                    st.info(f"**Diet:** {info['diet'][0]}")
                    st.info(f"**Prevention:** {info['prevention'][0]}")
                    st.warning(f"**Warning:** {info['warning_signs']}")

# ============================================
# TAB 3: Home Remedies
# ============================================
with tab3:
    st.subheader("🌿 Natural Home Remedies")
    
    remedy_categories = {
        "For Skin Issues": ["Acne", "Fungal infection", "Psoriasis", "Impetigo"],
        "For Respiratory": ["Common Cold", "Bronchial Asthma", "Pneumonia", "Tuberculosis"],
        "For Digestive": ["Gastroenteritis", "Peptic ulcer diseae", "GERD"],
        "For General": ["Migraine", "Fatigue", "Fever", "Headache"]
    }
    
    for category, diseases in remedy_categories.items():
        st.markdown(f"### {category}")
        cols = st.columns(3)
        for i, disease in enumerate(diseases):
            if disease in DISEASE_COMPLETE_INFO:
                with cols[i % 3]:
                    with st.expander(f"🌱 {disease}"):
                        info = DISEASE_COMPLETE_INFO[disease]
                        for remedy in info["home_remedies"][:3]:
                            st.markdown(remedy)

# ============================================
# TAB 4: Diet & Exercise
# ============================================
with tab4:
    st.subheader("🥗 Diet & Exercise Recommendations")
    
    selected = st.selectbox("Select a condition:", ALL_DISEASES)
    
    if selected:
        info = get_disease_info(selected)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥗 Recommended Diet")
            for diet in info["diet"]:
                st.markdown(f"• {diet}")
        
        with col2:
            st.markdown("### 🏃‍♂️ Exercise Guidelines")
            st.info(info["exercise"])
            st.markdown("### 🛡️ Prevention")
            for prev in info["prevention"][:3]:
                st.markdown(f"• {prev}")

# ============================================
# TAB 5: Health Tips
# ============================================
with tab5:
    st.subheader("📚 Daily Health Tips & Awareness")
    
    tip_types = st.radio("Select:", ["Daily Tips", "Warning Signs", "Prevention Guide", "Nutrition Facts"])
    
    if tip_types == "Daily Tips":
        tips_list = [
            "💧 **Hydration**: Drink water before you feel thirsty",
            "😴 **Sleep**: 7-8 hours for adults, 8-10 for teens",
            "🏃 **Movement**: Take 5-minute walk breaks every hour",
            "🧘 **Stress**: 10 minutes meditation daily",
            "🍎 **Diet**: Eat a rainbow of fruits and vegetables",
            "🧼 **Hygiene**: Wash hands for 20 seconds",
            "☀️ **Sun**: 15 minutes morning sunlight for Vitamin D",
            "📱 **Digital Detox**: No screens 1 hour before bed"
        ]
        for tip in tips_list:
            st.markdown(tip)
            st.markdown("---")
    
    elif tip_types == "Warning Signs":
        warnings_list = [
            "🚨 **Chest Pain + Shortness of Breath** → Seek ER immediately",
            "🚨 **Severe Headache + Confusion** → Possible stroke",
            "🚨 **Blood in Stool/Vomit** → Internal bleeding risk",
            "🚨 **High Fever (104°F+) + Stiff Neck** → Meningitis possible",
            "🚨 **Sudden Vision Loss** → Eye emergency",
            "🚨 **Difficulty Breathing** → Respiratory distress"
        ]
        for warn in warnings_list:
            st.warning(warn)
            st.markdown("---")
    
    elif tip_types == "Prevention Guide":
        st.markdown("""
        ### 🛡️ Disease Prevention Guide
        
        **1. Vaccination Schedule**
        - Annual flu shot
        - COVID-19 boosters
        - Pneumonia vaccine (65+)
        - HPV vaccine (9-45 years)
        
        **2. Lifestyle Habits**
        - No smoking, limit alcohol
        - Maintain healthy weight
        - Regular health screenings
        - Stress management
        
        **3. Hygiene Practices**
        - Hand washing
        - Cover coughs/sneezes
        - Don't share personal items
        - Clean high-touch surfaces
        """)
    
    else:
        st.markdown("""
        ### 🥗 Essential Nutrition Facts
        
        **Vitamin C** 🍊
        - Boosts immune system
        - Sources: Citrus, bell peppers, kiwi
        
        **Vitamin D** ☀️
        - Bone health, immunity
        - Sources: Sunlight, fatty fish, fortified milk
        
        **Zinc** 🥜
        - Wound healing, immune function
        - Sources: Nuts, seeds, legumes, meat
        
        **Omega-3** 🐟
        - Anti-inflammatory
        - Sources: Salmon, walnuts, flaxseeds
        
        **Probiotics** 🥛
        - Gut health, digestion
        - Sources: Yogurt, kefir, kimchi, kombucha
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🩺 <strong>AI Health Assistant</strong> | Powered by Machine Learning</p>
        <p style='font-size: 12px;'>⚠️ Educational purpose only. Always consult a healthcare provider.</p>
    </div>
    """,
    unsafe_allow_html=True
)
