# llm_helper.py
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_disease_info_from_llm(disease_name, feature_type="remedies"):
    """
    Get disease information using Groq LLM
    
    feature_type can be: "remedies", "diet", "exercise", "prevention", "awareness"
    """
    
    prompts = {
        "remedies": f"List 5 simple home remedies for {disease_name}. Keep each remedy short and practical. Format each line with •",
        "diet": f"List 5 dietary recommendations for someone with {disease_name}. Format each line with •",
        "exercise": f"Recommend safe exercises for someone with {disease_name}. Format each line with •",
        "prevention": f"List 5 prevention tips to avoid {disease_name}. Format each line with •",
        "awareness": f"Provide 3 important facts about {disease_name} that everyone should know about early detection and management."
    }
    
    prompt = prompts.get(feature_type, prompts["remedies"])
    
    if not GROQ_API_KEY:
        return f"⚠️ GROQ_API_KEY not found. Please set it in environment variables."
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful health assistant. Provide accurate, practical health information. Never give medical advice. Always encourage consulting healthcare providers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # Lower temperature for consistent, factual responses
            max_tokens=500
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error fetching information: {str(e)}\n\nPlease consult a healthcare provider for accurate medical advice."

def get_complete_health_advice(disease_name):
    """Get complete health advice from Groq LLM"""
    
    prompt = f"""As a health assistant, provide information about {disease_name} in this exact format:

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
[2-3 sentences about safe exercises]

📚 AWARENESS:
[2-3 important facts about early detection and management]

⚠️ IMPORTANT: Keep responses practical and educational. Always include disclaimer to consult healthcare provider."""
    
    if not GROQ_API_KEY:
        return f"⚠️ GROQ_API_KEY not found. Please set it in environment variables.\n\nPredicted disease: {disease_name}\n\nPlease consult a healthcare provider for medical advice."
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical information assistant. Provide accurate, helpful health information. Never give treatment advice. Always recommend consulting doctors."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # Consistent, factual responses
            max_tokens=800
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n**Predicted Disease:** {disease_name}\n\nPlease consult a healthcare provider for proper medical advice."

def get_symptom_analysis(symptoms_list, predicted_disease, confidence):
    """Get AI analysis of symptoms"""
    
    prompt = f"""A patient reported these symptoms: {', '.join(symptoms_list)}.
The AI predicted: {predicted_disease} with {confidence:.1f}% confidence.

Provide a brief analysis (2-3 sentences) explaining why these symptoms match this condition.
Keep it educational, not diagnostic."""
    
    if not GROQ_API_KEY:
        return f"Based on the symptoms reported, the AI model suggests {predicted_disease} with {confidence:.1f}% confidence."
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical educator. Explain symptom-disease relationships in simple terms."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"The AI model has identified {predicted_disease} as the most likely condition based on the symptoms reported."
