import streamlit as st
import os

st.title("Secret Test")

# Try to read secret
try:
    api_key = st.secrets.get("GROQ_API_KEY", "NOT FOUND")
    if api_key != "NOT FOUND":
        st.success(f"✅ API Key found! Length: {len(api_key)}")
        st.info(f"Starts with: {api_key[:10]}...")
    else:
        st.error("❌ API Key NOT found in secrets")
except Exception as e:
    st.error(f"Error: {e}")

# Show all secrets keys (not values)
st.write("Secret keys available:", list(st.secrets.keys()) if hasattr(st, 'secrets') else "No secrets")
