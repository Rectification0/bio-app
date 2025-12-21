import streamlit as st

st.set_page_config(
    page_title="NutriSense - Smart Soil Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import json
from datetime import datetime
from typing import Dict
from groq import Groq
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from pydantic import BaseModel, field_validator
import hashlib

class SoilData(BaseModel):
    pH: float
    EC: float
    Moisture: float
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Microbial: float
    Temperature: float
    
    @field_validator('pH')
    @classmethod
    def pH_range(cls, v):
        if not 0 <= v <= 14:
            raise ValueError('pH must be 0-14')
        return v

def get_health_score(soil: Dict) -> float:
    try:
        ph = max(0, 25 - abs(soil['pH'] - 7.0) * 3.5)
        ec = max(0, 25 - min(soil['EC'], 4.0) * 6.25)
        moist = 20 if 25 <= soil['Moisture'] <= 40 else max(0, 20 - abs(soil['Moisture'] - 32.5) * 0.5)
        npk = min(soil['Nitrogen']/80*10, 10) + min(soil['Phosphorus']/50*10, 10) + min(soil['Potassium']/250*10, 10)
        return min(max(ph + ec + moist + npk, 0), 100)
    except:
        return 50.0

def interpret(param: str, val: float) -> tuple:
    data = {
        'pH': [(0,5.5,"Acidic","🔴"),(5.5,6.5,"Low","🟡"),(6.5,7.5,"Optimal","🟢"),(7.5,8.5,"High","🟡"),(8.5,15,"Alkaline","🔴")],
        'EC': [(0,0.8,"Low","🟢"),(0.8,2,"Moderate","🟡"),(2,4,"High","🟠"),(4,25,"Very High","🔴")],
        'Moisture': [(0,15,"Dry","🔴"),(15,25,"Low","🟡"),(25,40,"Optimal","🟢"),(40,60,"High","🟡"),(60,101,"Wet","🔴")],
        'Nitrogen': [(0,40,"Low","🔴"),(40,80,"Optimal","🟢"),(80,501,"High","🟡")],
        'Phosphorus': [(0,20,"Low","🔴"),(20,50,"Optimal","🟢"),(50,201,"High","🟡")],
        'Potassium': [(0,100,"Low","🔴"),(100,250,"Optimal","🟢"),(250,501,"High","🟡")],
        'Microbial': [(0,3,"Poor","🔴"),(3,7,"Good","🟢"),(7,11,"Excellent","💚")],
        'Temperature': [(0,10,"Cold","🔵"),(10,30,"Optimal","🟢"),(30,51,"Hot","🔴")]
    }
    for low, high, status, emoji in data.get(param, []):
        if low <= val < high:
            return status, emoji
    return "Unknown", "⚪"

@st.cache_resource
def init_db():
    try:
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect('data/soil_history.db', check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS soil_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hash TEXT UNIQUE,
                soil_data TEXT,
                timestamp DATETIME,
                summary TEXT,
                location TEXT,
                health_score REAL
            )
        """)
        
        # Add health_score column if it doesn't exist (for existing databases)
        try:
            conn.execute("ALTER TABLE soil_records ADD COLUMN health_score REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        conn.commit()
        return conn
    except Exception as e:
        st.error(f"DB error: {e}")
        return None

@st.cache_resource
def get_groq_client():
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        return Groq(api_key=api_key) if api_key else None
    except:
        return None

def build_prompt(soil: Dict, task: str, loc: str = "") -> str:
    base = f"""Soil Data{f' - {loc}' if loc else ''}:
pH: {soil['pH']:.2f}, EC: {soil['EC']:.2f} dS/m, Moisture: {soil['Moisture']:.1f}%
N: {soil['Nitrogen']:.2f}, P: {soil['Phosphorus']:.2f}, K: {soil['Potassium']:.2f} mg/kg
Microbial: {soil['Microbial']:.2f}/10, Temp: {soil['Temperature']:.1f}°C"""

    prompts = {
        "summary": f"{base}\n\nProvide: 1) Overall condition 2) Main concerns 3) Top 3 actions. Keep brief.",
        "crops": f"{base}\n\nSuggest TOP 5 suitable crops with reasons. Include Indian varieties.",
        "fertilizer": f"{base}\n\nProvide: NPK ratio, kg/hectare, timing, organic alternatives.",
        "irrigation": f"{base}\n\nProvide: frequency, water amount, best timing for irrigation."
    }
    return prompts.get(task, base)

@st.cache_data(ttl=300)
def call_groq(_hash: str, prompt: str, _task: str) -> str:
    client = get_groq_client()
    if not client:
        return "⚠️ Configure GROQ_API_KEY"
    
    model = st.session_state.get("selected_model", "llama-3.3-70b-versatile")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an agricultural expert. Provide practical advice for Indian farmers in simple language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"

def save_record(soil: Dict, summary: str, loc: str = ""):
    try:
        conn = init_db()
        if conn:
            data_str = json.dumps(soil)
            hash_val = hashlib.md5(data_str.encode()).hexdigest()
            health_score = get_health_score(soil)
            conn.execute(
                "INSERT OR IGNORE INTO soil_records (data_hash, soil_data, timestamp, summary, location, health_score) VALUES (?,?,?,?,?,?)",
                (hash_val, data_str, datetime.now(), summary, loc, health_score)
            )
            conn.commit()
    except:
        pass

def load_history() -> pd.DataFrame:
    try:
        conn = init_db()
        if conn:
            return pd.read_sql_query("SELECT * FROM soil_records ORDER BY timestamp DESC LIMIT 30", conn)
    except:
        pass
    return pd.DataFrame()

# Init session
for key in ['soil_data', 'location', 'selected_model']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'soil_data' else ("" if key == 'location' else "llama-3.3-70b-versatile")

# CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.metric-card {
    background: #f5f7fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}
.status-good {
    border-left: 4px solid #10b981;
    background: #d1fae5;
    padding: 0.8rem;
    border-radius: 5px;
    margin: 0.3rem 0;
}
.status-warning {
    border-left: 4px solid #f59e0b;
    background: #fef3c7;
    padding: 0.8rem;
    border-radius: 5px;
    margin: 0.3rem 0;
}
.status-critical {
    border-left: 4px solid #ef4444;
    background: #fee2e2;
    padding: 0.8rem;
    border-radius: 5px;
    margin: 0.3rem 0;
}
.rec-box {
    background: #f0f9ff;
    border-left: 4px solid #3b82f6;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🌱 NutriSense - Smart Soil Analytics</h1><p>IoT-Enabled Precision Agriculture System</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System")
    st.success("✅ AI Online" if get_groq_client() else "⚠️ Configure API")
    st.success("✅ DB Connected" if init_db() else "❌ DB Error")
    
    st.divider()
    
    st.header("🤖 AI Model")
    models = {
        "Llama 3.3 70B": "llama-3.3-70b-versatile",
        "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant", 
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma 2 9B": "gemma2-9b-it"
    }
    selected = st.selectbox("Model", list(models.keys()), index=0)
    st.session_state.selected_model = models[selected]
    
    st.divider()
    
    history = load_history()
    if not history.empty:
        st.metric("Analyses", len(history))
        if 'health_score' in history.columns and history['health_score'].notna().any():
            avg_score = history['health_score'].dropna().mean()
            st.metric("Avg Score", f"{avg_score:.0f}/100")
        if st.button("🗑️ Clear History", use_container_width=True):
            conn = init_db()
            if conn:
                conn.execute("DELETE FROM soil_records")
                conn.commit()
                st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "➕ Input", "📈 History", "📚 Guide"])

with tab1:
    if not st.session_state.soil_data:
        st.info("👉 Enter soil data in **Input** tab")
        st.markdown("""
        ### Features
        - 🧪 Chemistry: pH, EC, NPK
        - 💧 Physical: Moisture
        - 🦠 Biological: Microbial activity
        - 🤖 AI recommendations
        """)
    else:
        soil = st.session_state.soil_data
        loc = st.session_state.location
        
        st.subheader(f"🎯 Overview{f' - {loc}' if loc else ''}")
        
        health = get_health_score(soil)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Health Score", f"{health:.0f}/100")
            st.success("Excellent" if health >= 70 else ("Good" if health >= 50 else "Needs work"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Parameters", "8")
            st.info("All tracked")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Last Update", "Just now")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("🔬 Parameters")
            params = {
                'pH': (soil['pH'], 'pH'), 'EC': (soil['EC'], 'dS/m'),
                'Moisture': (soil['Moisture'], '%'), 'Nitrogen': (soil['Nitrogen'], 'mg/kg'),
                'Phosphorus': (soil['Phosphorus'], 'mg/kg'), 'Potassium': (soil['Potassium'], 'mg/kg'),
                'Microbial': (soil['Microbial'], 'Index'), 'Temperature': (soil['Temperature'], '°C')
            }
            
            for name, (val, unit) in params.items():
                status, emoji = interpret(name, val)
                css = "status-good" if "🟢" in emoji or "💚" in emoji else ("status-warning" if "🟡" in emoji else "status-critical")
                st.markdown(f'<div class="{css}"><b>{emoji} {name}:</b> {val:.1f} {unit} - {status}</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Visuals")
            
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health,
                title={'text': "Health Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 40], 'color': "#fee2e2"},
                        {'range': [40, 70], 'color': "#fef3c7"},
                        {'range': [70, 100], 'color': "#d1fae5"}
                    ]
                }
            ))
            fig1.update_layout(height=250)
            st.plotly_chart(fig1, use_container_width=True)
        
        st.divider()
        
        st.subheader("🤖 AI Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✨ Summary", use_container_width=True, type="primary"):
                with st.spinner("Analyzing..."):
                    prompt = build_prompt(soil, "summary", loc)
                    result = call_groq(hashlib.md5(prompt.encode()).hexdigest(), prompt, "summary")
                    st.session_state.summary = result
                    save_record(soil, result, loc)
                    st.rerun()
        
        with col2:
            if st.button("🌾 Crops", use_container_width=True):
                with st.spinner("Finding crops..."):
                    prompt = build_prompt(soil, "crops", loc)
                    st.session_state.crops = call_groq(hashlib.md5(prompt.encode()).hexdigest(), prompt, "crops")
                    st.rerun()
        
        with col3:
            if st.button("💊 Fertilizer", use_container_width=True):
                with st.spinner("Calculating..."):
                    prompt = build_prompt(soil, "fertilizer", loc)
                    st.session_state.fertilizer = call_groq(hashlib.md5(prompt.encode()).hexdigest(), prompt, "fertilizer")
                    st.rerun()
        
        if 'summary' in st.session_state and st.session_state.summary:
            st.markdown(f'<div class="rec-box"><h4>📋 Summary</h4>{st.session_state.summary}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'crops' in st.session_state and st.session_state.crops:
                st.markdown(f'<div class="rec-box"><h4>🌾 Crops</h4>{st.session_state.crops}</div>', unsafe_allow_html=True)
        
        with col2:
            if 'fertilizer' in st.session_state and st.session_state.fertilizer:
                st.markdown(f'<div class="rec-box"><h4>💊 Fertilizer</h4>{st.session_state.fertilizer}</div>', unsafe_allow_html=True)

with tab2:
    st.header("➕ Enter Soil Data")
    
    loc_input = st.text_input("📍 Location (Optional)", value=st.session_state.location, placeholder="e.g., Field A")
    
    with st.form("soil_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧪 Chemical")
            pH = st.number_input("pH", 0.0, 14.0, 7.0, 0.1, help="Optimal: 6.5-7.5")
            EC = st.number_input("EC (dS/m)", 0.0, 20.0, 2.0, 0.1, help="<2.0 ideal")
            N = st.number_input("Nitrogen (mg/kg)", 0.0, 500.0, 50.0, 1.0, help="40-80 optimal")
            P = st.number_input("Phosphorus (mg/kg)", 0.0, 200.0, 30.0, 1.0, help="20-50 optimal")
        
        with col2:
            st.subheader("🌡️ Physical")
            Moist = st.number_input("Moisture (%)", 0.0, 100.0, 25.0, 1.0)
            Temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
            K = st.number_input("Potassium (mg/kg)", 0.0, 500.0, 150.0, 1.0, help="100-250 optimal")
            Micro = st.number_input("Microbial Index", 0.0, 10.0, 5.0, 0.1, help="0-10 scale")
        
        if st.form_submit_button("💾 Save & Analyze", use_container_width=True, type="primary"):
            try:
                soil_dict = {
                    "pH": pH, "EC": EC, "Moisture": Moist, "Nitrogen": N, 
                    "Phosphorus": P, "Potassium": K, "Microbial": Micro, "Temperature": Temp
                }
                
                SoilData(**soil_dict)
                st.session_state.soil_data = soil_dict
                st.session_state.location = loc_input
                st.session_state.timestamp = datetime.now().isoformat()
                
                for key in ['summary', 'crops', 'fertilizer']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("✅ Saved! Go to Dashboard")
                st.balloons()
                
            except ValueError as e:
                st.error(f"❌ Invalid: {e}")

with tab3:
    st.header("📈 History")
    
    history = load_history()
    if not history.empty:
        st.success(f"📊 {len(history)} records")
        
        for _, row in history.iterrows():
            with st.expander(f"📅 {row['timestamp']} - {row.get('location', 'Unknown')} - Score: {row.get('health_score', 'N/A'):.0f}/100" if row.get('health_score') else f"📅 {row['timestamp']} - {row.get('location', 'Unknown')}"):
                try:
                    data = json.loads(row['soil_data'])
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("pH", f"{data.get('pH', 0):.1f}")
                        st.metric("N", f"{data.get('Nitrogen', 0):.0f}")
                    with col2:
                        st.metric("EC", f"{data.get('EC', 0):.2f}")
                        st.metric("P", f"{data.get('Phosphorus', 0):.0f}")
                    with col3:
                        st.metric("Moisture", f"{data.get('Moisture', 0):.0f}")
                        st.metric("K", f"{data.get('Potassium', 0):.0f}")
                    
                    if row.get('health_score'):
                        st.info(f"🎯 Health Score: {row['health_score']:.1f}/100")
                    
                    if row['summary']:
                        st.info(row['summary'])
                    
                    if st.button(f"🔄 Load", key=f"load_{row['id']}"):
                        st.session_state.soil_data = data
                        st.session_state.location = row.get('location', '')
                        st.success("Loaded!")
                        st.rerun()
                except:
                    st.error("Error loading")
    else:
        st.info("No history")

with tab4:
    st.header("📚 Knowledge Base")
    
    with st.expander("🎯 pH - Acidity/Alkalinity"):
        st.markdown("""
        **Optimal:** 6.5-7.5
        - **Low (<5.5):** Nutrient lockout, aluminum toxicity
        - **High (>8.5):** Iron/zinc deficiency
        **Fix:** Lime for low, sulfur for high
        """)
    
    with st.expander("⚡ EC - Salinity"):
        st.markdown("""
        **Optimal:** <0.8 dS/m
        - **High EC:** Reduces water uptake
        **Fix:** Leaching, drainage
        """)
    
    with st.expander("🌿 NPK Nutrients"):
        st.markdown("""
        **N (40-80):** Leaf growth - Urea
        **P (20-50):** Roots, flowers - DAP
        **K (100-250):** Disease resistance - MOP
        """)
    
    with st.expander("💧 Moisture"):
        st.markdown("""
        **Optimal:** 25-40%
        - Low: Drought stress
        - High: Root rot
        """)
    
    with st.expander("🦠 Microbial Activity"):
        st.markdown("""
        **Good:** 5-7/10
        Improves nutrients & structure
        **Boost:** Add compost, reduce tillage
        """)
    
    st.success("💡 Test soil every 3-6 months for best results!")