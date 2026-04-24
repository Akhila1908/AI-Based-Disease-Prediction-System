# llm_helper.py
import os
import streamlit as st
from groq import Groq

# Get API key from Streamlit secrets or environment variable
def get_groq_client():
    """Initialize Groq client with proper error handling"""
    try:
        # Try to get from Streamlit secrets first
        api_key = st.secrets.get("GROQ_API_KEY")
        
        # If not in secrets, try environment variable
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            return None, "GROQ_API_KEY not found. Please add it to Streamlit secrets or environment variables."
        
        # Initialize client without proxies parameter
        client = Groq(api_key=api_key)
        return client, None
        
    except TypeError as e:
        if "proxies" in str(e):
            # Version compatibility issue - try alternative initialization
            try:
                # Alternative initialization for older versions
                client = Groq(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                return client, None
            except Exception as e2:
                return None, f"Failed to initialize Groq client: {str(e2)}"
        else:
            return None, f"Failed to initialize Groq client: {str(e)}"
    except Exception as e:
        return None, f"Failed to initialize Groq client: {str(e)}"

def get_disease_info_from_llm(disease_name, feature_type="remedies"):
    """
    Get disease information using Groq LLM
    """
    prompts = {
        "remedies": f"List 5 simple home remedies for {disease_name}. Keep each remedy short and practical. Format each line with •",
        "diet": f"List 5 dietary recommendations for someone with {disease_name}. Format each line with •",
        "exercise": f"Recommend safe exercises for someone with {disease_name}. Format each line with •",
        "prevention": f"List 5 prevention tips to avoid {disease_name}. Format each line with •",
        "awareness": f"Provide 3 important facts about {disease_name} that everyone should know about early detection and management."
    }
    
    prompt = prompts.get(feature_type, prompts["remedies"])
    
    client, error = get_groq_client()
    if error:
        return f"⚠️ {error}"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful health assistant. Provide accurate, practical health information. Never give medical advice. Always encourage consulting healthcare providers. Keep responses concise and factual."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease consult a healthcare provider for accurate medical advice."

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
    
    client, error = get_groq_client()
    if error:
        return f"⚠️ {error}\n\n**Predicted Disease:** {disease_name}\n\nPlease consult a healthcare provider for medical advice."
    
    try:
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
            temperature=0.0,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n**Predicted Disease:** {disease_name}\n\nPlease consult a healthcare provider for proper medical advice."

def get_symptom_analysis(symptoms_list, predicted_disease, confidence):
    """Get AI analysis of symptoms"""
    
    prompt = f"A patient reported these symptoms: {', '.join(symptoms_list)}.\nThe AI predicted: {predicted_disease} with {confidence:.1f}% confidence.\n\nProvide a brief analysis (2-3 sentences) explaining why these symptoms match this condition.\nKeep it educational, not diagnostic."
    
    client, error = get_groq_client()
    if error:
        return f"The AI model has identified {predicted_disease} as the most likely condition based on the symptoms reported with {confidence:.1f}% confidence."
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical educator. Explain symptom-disease relationships in simple, educational terms. Never diagnose or give treatment advice."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=250
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"The AI model has identified {predicted_disease} as the most likely condition based on the symptoms reported with {confidence:.1f}% confidence."
