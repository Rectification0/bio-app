import os
import json
import streamlit as st
from datetime import datetime
from groq import Groq

st.set_page_config(page_title="NutriSense", layout="wide")

if "current_soil" not in st.session_state:
    st.session_state.current_soil = None

if "soil_history" not in st.session_state:
    st.session_state.soil_history = []

if "groq_client" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.session_state.groq_client = Groq(api_key=api_key)
    else:
        st.session_state.groq_client = None

st.title("🌱 NutriSense – Soil Intelligence Platform")

with st.sidebar:
    st.header("System Status")
    if st.session_state.current_soil:
        st.success("Soil data loaded")
        st.write("Last updated:")
        st.write(st.session_state.current_soil["timestamp"])
    else:
        st.warning("No soil data available")
    st.divider()
    st.write("Phase-1: Manual Input + AI Reasoning")
    st.write("Backend LLM: Groq")

def call_groq(prompt):
    client = st.session_state.groq_client
    if client is None:
        return "Groq API key not configured."
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an agronomy expert giving concise, actionable advice."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=600
    )
    return completion.choices[0].message.content

st.subheader("Quick Actions")

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate AI Soil Summary"):
        if st.session_state.current_soil:
            soil = st.session_state.current_soil
            prompt = f"""
            Soil data:
            pH: {soil['ph']}
            EC: {soil['ec']}
            Moisture: {soil['moisture']}
            Nitrogen: {soil['n']}
            Phosphorus: {soil['p']}
            Potassium: {soil['k']}
            Microbial Index: {soil['microbial']}
            Temperature: {soil['temperature']}
            Give a short soil health summary and main risks.
            """
            with st.spinner("Analyzing soil..."):
                result = call_groq(prompt)
            st.markdown(result)
        else:
            st.error("No soil data available.")

with col2:
    if st.button("Suggest Suitable Crops"):
        if st.session_state.current_soil:
            soil = st.session_state.current_soil
            prompt = f"""
            Based on the following soil data, suggest 3 suitable crops and brief reasons:
            pH: {soil['ph']}
            EC: {soil['ec']}
            Moisture: {soil['moisture']}
            Nitrogen: {soil['n']}
            Phosphorus: {soil['p']}
            Potassium: {soil['k']}
            Microbial Index: {soil['microbial']}
            """
            with st.spinner("Generating crop suggestions..."):
                result = call_groq(prompt)
            st.markdown(result)
        else:
            st.error("No soil data available.")

st.divider()

st.subheader("Current Soil Snapshot")

if st.session_state.current_soil:
    st.json(st.session_state.current_soil)
else:
    st.info("Enter soil data from the Input page to begin.")
