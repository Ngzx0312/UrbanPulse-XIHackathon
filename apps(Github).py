import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import pickle
import time
import ollama
import os
from groq import Groq
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="UrbanPulse: GovDSS",
    page_icon="🏙️",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL UI STYLING (CSS) ---
st.markdown("""
<style>
    /* 1. MAIN BACKGROUND & FONTS */
    .stApp {
        background-color: #FFFDFA;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* 2. CUSTOM SIDEBAR styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* 3. METRIC CARDS */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #58A6FF;
    }
    
    /* 4. BUTTONS*/
    div.stButton > button {
        width: 100%;
        height: 60px;
        border-radius: 8px;
        border: 1px solid #30363D;
        background-color: #21262D;
        color: #C9D1D9;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #30363D;
        color: #58A6FF;
        border-color: #58A6FF;
    }
    div.stButton > button:active {
        background-color: #1F6FEB;
        color: white;
    }

    /* 5. CUSTOM HEADERS */
    .header-style {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 20px;
        border-bottom: 2px solid #30363D;
        padding-bottom: 10px;
    }
    
    /* 6. STATUS BADGES */
    .status-live {
        background-color: #238636;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        vertical-align: middle;
        margin-left: 10px;
    }
    
    /* 7. AI CHAT BOX */
    .ai-box {
        background-color: #1F2937; 
        border-left: 4px solid #8B5CF6; 
        padding: 15px;
        border-radius: 5px;
        color: #E5E7EB;
        font-size: 14px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA & CONFIG ---
CLIMATE_DATA = {"Annual_Rainy_Days": 210, "Rain_Probability": 0.57}

LOCATIONS = {
    "Jalan Tun Razak (Commercial)": {
        "id": "Tun_Razak", "coords": [3.1579, 101.7116], "district": "Bukit Bintang", 
        "density": 11000, "base_traffic": 8500, "vulnerable_pop": 0.12
    },
    "Bangsar South (Mixed Usage)": {
        "id": "Bangsar", "coords": [3.1110, 101.6650], "district": "Bangsar", 
        "density": 6500, "base_traffic": 6000, "vulnerable_pop": 0.08
    },
    "Cheras Utama (Residential)": {
        "id": "Cheras", "coords": [3.0550, 101.7560], "district": "Cheras", 
        "density": 9500, "base_traffic": 7000, "vulnerable_pop": 0.15
    }
}

# --- 4. ENGINES ---
# --- 3. MATH ENGINE (With Self-Training Capability) ---
class UrbanPulseAI:
    def __init__(self):
        # Check if models exist. If not, TRAIN THEM NOW.
        if not os.path.exists("model_stress.pkl"):
            self.emergency_train()
            
        try:
            with open("model_stress.pkl", "rb") as f: self.stress_model = pickle.load(f)
            with open("model_roi.pkl", "rb") as f: self.roi_model = pickle.load(f)
            with open("encoder.pkl", "rb") as f: self.le = pickle.load(f)
            self.loaded = True
        except:
            self.loaded = False

    def emergency_train(self):
        """
        Hackathon Fix: Trains a lightweight model on the fly if files are missing.
        This allows deployment without uploading large .pkl files.
        """
        print("⚠️ Models missing. Initiating Emergency Training Protocol...")
        
        # 1. Generate Lightweight Data (1000 samples is enough for a demo)
        data = []
        districts = {
            "Bukit Bintang": {"density": 11000, "base_traffic": 8500},
            "Cheras":        {"density": 9500,  "base_traffic": 7000},
            "Bangsar":       {"density": 6500,  "base_traffic": 5500},
        }
        
        for _ in range(1000):
            dist_name = np.random.choice(list(districts.keys()))
            profile = districts[dist_name]
            weather = np.random.choice([0, 1, 2]) 
            intervention = np.random.randint(0, 6)
            
            # Simple Logic for Training
            traffic = profile["base_traffic"]
            if weather == 0: traffic *= 0.9
            if intervention == 2: traffic *= 0.85
            if intervention == 5: traffic *= 0.70
            
            stress = (traffic/10000 * 50) + 30
            if intervention == 1: stress -= 10
            if intervention == 4: stress += 15
            
            roi = 0.5
            if intervention == 1: roi = 1.2
            if intervention == 5: roi = 4.5
            
            data.append([dist_name, traffic, weather, intervention, profile["density"], stress, roi])
            
        df = pd.DataFrame(data, columns=["district", "traffic", "weather", "intervention", "density", "stress", "roi"])
        
        # 2. Train
        le = LabelEncoder()
        df['district_code'] = le.fit_transform(df['district'])
        X = df[["traffic", "weather", "intervention", "density", "district_code"]]
        
        m_stress = RandomForestRegressor(n_estimators=10).fit(X, df["stress"]) # Low estimators for speed
        m_roi = RandomForestRegressor(n_estimators=10).fit(X, df["roi"])
        
        # 3. Save
        with open("model_stress.pkl", "wb") as f: pickle.dump(m_stress, f)
        with open("model_roi.pkl", "wb") as f: pickle.dump(m_roi, f)
        with open("encoder.pkl", "wb") as f: pickle.dump(le, f)
        print("✅ Emergency Training Complete.")

    def predict(self, loc_data, intervention_code):
        if not self.loaded: return 0,0,0, {}
        i_code_map = {"Trees": 1, "Bike": 2, "Emergency": 3, "Flyover": 4, "PublicTransport": 5}
        i_code = i_code_map.get(intervention_code, 0)
        
        # Weighted Climate Logic
        t_sunny = loc_data["base_traffic"] + 500
        if i_code == 2: t_sunny *= 0.85
        if i_code == 5: t_sunny *= 0.70
        t_rain = loc_data["base_traffic"]
        if i_code == 2: t_rain *= 0.98
        if i_code == 5: t_rain *= 0.75
        avg_traffic = (t_sunny * 0.43) + (t_rain * 0.57)
        if i_code == 4: avg_traffic *= 1.15 # Induced demand

        # ML Prediction
        ml_i_code = i_code if i_code <= 3 else 0 
        try: dist_code = self.le.transform([loc_data["district"]])[0]
        except: dist_code = 0
        
        inputs = pd.DataFrame([[avg_traffic, 1, ml_i_code, loc_data["density"], dist_code]], 
                              columns=["traffic", "weather", "intervention", "density", "district_code"])
        stress = self.stress_model.predict(inputs)[0]
        roi = self.roi_model.predict(inputs)[0]
        
        # Manual Adjustments
        if i_code == 4: stress += 15; roi -= 1.5
        if i_code == 5: stress -= 10; roi += 4.5
        
        breakdown = {"Healthcare (Asthma)": roi * 0.6, "Productivity": roi * 0.3, "Fuel Savings": roi * 0.1}
        return round(stress, 1), round(roi, 2), int(avg_traffic), breakdown

# --- 4. SEA_LION AI COPILOT (Cloud API Version) ---
class SeaLionBrain:
    def __init__(self):
        # 1. SETUP CLIENT
        self.api_key = "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" 
        
        try:
            self.client = Groq(api_key=self.api_key)
            self.online = True
        except:
            self.online = False

    def ask_copilot(self, location, intervention, stress_score, roi, weather):

        # 2. PROMPT ENGINEERING (The "Persona" Injection)
        system_prompt = """
        ROLE: You are 'SEA-LION', a Sovereign AI developed by AI Singapore. 
        JOB: Senior Town Planner for DBKL (Kuala Lumpur City Hall).
        TONE: Formal Malaysian Government style (Bahasa Baku mixed with Professional English).
        
        INSTRUCTIONS:
        1. Review the urban simulation data provided.
        2. Reference local context (e.g., 'Rancangan Struktur KL 2040', 'Musim Tengkujuh').
        3. Keep response under 3 sentences.
        4. If the intervention is 'Flyover', warn about 'Induced Demand'.
        """
        
        user_prompt = f"""
        DATA LAPORAN:
        - Lokasi: {location}
        - Cuaca: {weather}
        - Intervensi: {intervention}
        - Skor Stres: {stress_score}/100
        - ROI: RM {roi} Juta
        
        Berikan ulasan teknikal ringkas.
        """

        try:
            # 3. CALL API 
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile", 
                temperature=0.3,       
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"⚠️ API Error: {str(e)}"

math_engine = UrbanPulseAI()
ai_brain = SeaLionBrain()

# --- 5. LAYOUT & LOGIC ---
if 'active_tool' not in st.session_state: st.session_state.active_tool = "None"

# HEADER ROW
st.image("banner.jpeg", use_container_width=True)
st.markdown("""
    <div style='display: flex; align-items: center;'>
        <h1 style='margin: 0; padding: 0;'>UrbanPulse <span style='color: #58A6FF;'>GovDSS</span></h1>
        <span class='status-live'>● SYSTEM ONLINE</span>
    </div>
    <p style='margin: 0; opacity: 0.7;'>Integrated City Planning & Health Impact Simulation Engine</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# MAIN WORKSPACE
col_sidebar, col_main = st.columns([1, 3])

# --- LEFT PANEL (CONTROLS) ---
with col_sidebar:
    st.markdown("### 📍 Project Context")
    selected_loc_name = st.selectbox("Select District", list(LOCATIONS.keys()), label_visibility="collapsed")
    loc_data = LOCATIONS[selected_loc_name]
    
    # District Stats Card
    st.info(f"""
    **District:** {loc_data['district']}
    **Density:** {loc_data['density']:,} /km²
    **Base Traffic:** {loc_data['base_traffic']:,} pcu/hr
    """)
    
    st.markdown("### 🛠️ Intervention Dock")
    
    # 2x3 Grid Buttons
    r1c1, r1c2 = st.columns(2)
    with r1c1: 
        if st.button("🚫 Clear"): st.session_state.active_tool = "None"
        if st.button("🚴 Bike Lane"): st.session_state.active_tool = "Bike"
        if st.button("🛣️ Flyover"): st.session_state.active_tool = "Flyover"
    with r1c2:
        if st.button("🌳 Green Way"): st.session_state.active_tool = "Trees"
        if st.button("🏥 EMS Route"): st.session_state.active_tool = "Emergency"
        if st.button("🚌 Transit"): st.session_state.active_tool = "PublicTransport"
        
    st.markdown("---")
    st.caption("Data Sources: MetMalaysia, OpenDOSM, MOH")

# --- RIGHT PANEL (VISUALIZATION) ---
with col_main:
    curr = st.session_state.active_tool
    
    # Run Simulation
    if curr == "None":
        s, r, t, bd = 85.0, 0.0, loc_data["base_traffic"], {}
        ai_msg = "Ready for simulation inputs."
    else:
        # Use a toast for slick feedback
        if 'last_run' not in st.session_state or st.session_state.last_run != curr:
            st.toast(f"Simulating Impact: {curr}...", icon="🔄")
            st.session_state.last_run = curr
            
        s, r, t, bd = math_engine.predict(loc_data, curr)
        ai_msg = ai_brain.ask_copilot(selected_loc_name, curr, s, r, "Tropical (Annual Weighted Avg)") if math_engine.loaded else "Models missing."

    # KPIS ROW (Styled Cards)
    k1, k2, k3 = st.columns(3)
    k1.metric("Community Stress", f"{s}/100", delta="-Score (Better)" if s<85 else "+Score (Worse)", delta_color="inverse")
    k2.metric("Traffic Volume", f"{t} /hr", delta="Annual Avg")
    k3.metric("Health ROI", f"RM {r} M", delta="Annual Savings")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # MAP & INTELLIGENCE SPLIT
    tab_map, tab_detail = st.tabs(["🗺️ Digital Twin Map", "📊 Deep Dive Analysis"])
    
    with tab_map:
        mc1, mc2 = st.columns([2.5, 1])
        with mc1:
            lat, lon = loc_data["coords"]
            # Color logic: Red if bad, Green/Blue if good
            color = [255, 50, 50, 180] if s > 70 else [0, 200, 100, 180]
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14.5, pitch=50),
                layers=[pdk.Layer("ScatterplotLayer", data=pd.DataFrame({'lat':[lat],'lon':[lon]}), get_position='[lon,lat]', get_color=color, get_radius=500, pickable=True)],
                map_style=pdk.map_styles.CARTO_DARK
            ), key=selected_loc_name)
        
        with mc2:
            st.markdown("#### 🤖 AI Policy Audit")
            st.markdown(f"<div class='ai-box'><b>Analisis SEA-LION:</b><br>{ai_msg}</div>", unsafe_allow_html=True)
            if curr == "Flyover":
                st.error("⚠️ **Warning:** Induced Demand Detected.")
    
    with tab_detail:
        c_dem, c_fin = st.columns(2)
        with c_dem:
            st.markdown("#### 👥 Demographic Impact")
            affected = int(loc_data["density"] * 1.5)
            vuln = int(affected * loc_data["vulnerable_pop"])
            st.bar_chart(pd.DataFrame({"Group": ["Adults", "Vulnerable (Kids/Elderly)"], "Count": [affected-vuln, vuln]}).set_index("Group"), color="#58A6FF")
        
        with c_fin:
            st.markdown("#### 💰 ROI Ledger")
            if r > 0:
                df_fin = pd.DataFrame(list(bd.items()), columns=["Category", "RM Millions"])
                st.dataframe(df_fin, hide_index=True, use_container_width=True)
            else:
                st.warning("Project ROI is negative/neutral.")
