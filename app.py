# ==============================
# IMPORTS
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import gdown
import time
import os
import pydeck as pdk

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="SentinelNet IDS", layout="wide")

# ==============================
# 🔥 MATRIX UI STYLE
# ==============================
st.markdown("""
<style>
body {
    background-color: black;
    color: #00ffcc;
    font-family: monospace;
}
h1 {
    text-shadow: 0 0 20px #00f7ff, 0 0 40px #00f7ff;
}
.stMetric {
    background: #020617;
    border: 1px solid #00f7ff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(0,255,255,0.5);
}
.stButton>button {
    background: black;
    border: 1px solid #00f7ff;
    color: #00f7ff;
    border-radius: 10px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #00f7ff;
    box-shadow: 0 0 15px rgba(0,255,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# ALERT SOUND
# ==============================
def play_alert():
    st.markdown("""
    <audio autoplay>
        <source src="https://www.soundjay.com/button/beep-07.wav">
    </audio>
    """, unsafe_allow_html=True)

# ==============================
# GOOGLE DRIVE LOADER
# ==============================
def load_pkl_from_drive(file_id, filename):
    if not os.path.exists(filename):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
    return joblib.load(filename)

def load_csv_from_drive(file_id, filename):
    if not os.path.exists(filename):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
    return pd.read_csv(filename)

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    iso = joblib.load("iso_model.pkl")
    ocsvm = joblib.load("ocsvm_model.pkl")
    pca = joblib.load("pca_model.pkl")
    t = joblib.load("threshold.pkl")
    lof = load_pkl_from_drive("1WcOnd7FWSw3fHUNVS-7jmEWI2-ggWWfY", "lof_model.pkl")
    return iso, ocsvm, pca, t, lof

iso, ocsvm, pca_model, best_t, lof = load_models()

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    synthetic_df = load_csv_from_drive(
        "1suz79KQkN4sbJT5n_B4vW844z06Lx04Y",
        "synthetic_network_data.csv"
    )
    X_real = load_pkl_from_drive(
        "1sLgVZFuHgyOlzbdnXhnEkCbEH6GcBk2E",
        "X_real.pkl"
    )
    y_real = joblib.load("y_real.pkl")
    return synthetic_df.values, X_real, y_real

X_synthetic_scaled, X_real, y_real = load_data()

# ==============================
# SESSION STATE
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# HEADER
# ==============================
st.markdown("""
<h1 style='text-align:center; color:#00f7ff;'>🚨 SentinelNet IDS</h1>
<p style='text-align:center; color:#aaa;'>AI-Powered Intrusion Detection System</p>
<hr>
""", unsafe_allow_html=True)

# ==============================
# SIMULATION
# ==============================
def simulate(sample_size):
    idx = np.random.choice(len(X_synthetic_scaled), sample_size, replace=False)
    X = X_synthetic_scaled[idx]

    iso_s = -iso.decision_function(X)
    lof_s = -lof.decision_function(X)

    X_pca = pca_model.transform(X)
    ocsvm_s = -ocsvm.decision_function(X_pca)

    iso_s = MinMaxScaler().fit_transform(iso_s.reshape(-1,1)).ravel()
    lof_s = MinMaxScaler().fit_transform(lof_s.reshape(-1,1)).ravel()
    ocsvm_s = MinMaxScaler().fit_transform(ocsvm_s.reshape(-1,1)).ravel()

    score = 0.6*iso_s + 0.25*ocsvm_s + 0.15*lof_s
    pred = (score >= best_t).astype(int)

    return pred, score, idx

# ==============================
# REAL EVALUATION
# ==============================
@st.cache_data
def evaluate_real():
    iso_s = -iso.decision_function(X_real)
    lof_s = -lof.decision_function(X_real)

    X_pca = pca_model.transform(X_real)
    ocsvm_s = -ocsvm.decision_function(X_pca)

    iso_s = MinMaxScaler().fit_transform(iso_s.reshape(-1,1)).ravel()
    lof_s = MinMaxScaler().fit_transform(lof_s.reshape(-1,1)).ravel()
    ocsvm_s = MinMaxScaler().fit_transform(ocsvm_s.reshape(-1,1)).ravel()

    score = 0.6*iso_s + 0.25*ocsvm_s + 0.15*lof_s
    pred = (score >= best_t).astype(int)

    return pred, score

# ==============================
# CONTROLS
# ==============================
mode = st.radio("Mode", ["Manual", "Real-Time"], horizontal=True)
sample_size = st.slider("Sample Size", 100, 5000, 1000)
run = st.button("🚀 Run Detection")

# ==============================
# RUN
# ==============================
if run or mode == "Real-Time":

    pred, scores, idx = simulate(sample_size)

    attack_count = int(np.sum(pred))
    normal_count = int(np.sum(pred == 0))
    attack_percent = float(np.mean(pred) * 100)

    st.session_state.history.append(attack_percent)

    # ALERT
    if attack_count > 0:
        play_alert()
        st.error(f"🚨 {attack_count} Intrusions Detected!")
    else:
        st.success("✅ SYSTEM SECURE")

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("Traffic", len(pred))
    c2.metric("Attacks", attack_count)
    c3.metric("Attack %", f"{attack_percent:.2f}%")

    st.markdown("---")

    # TREND
    st.subheader("📈 Attack Trend")
    st.line_chart(pd.DataFrame({"Attack %": st.session_state.history}))

    # ==============================
    # 🌍 PYDECK CYBER MAP
    # ==============================
    st.subheader("🌍 Real-Time Cyber Attack Map")

    regions = {
        "USA": (37.77, -122.41),
        "India": (28.61, 77.20),
        "UK": (51.50, -0.12),
        "Germany": (52.52, 13.40),
        "Russia": (55.75, 37.61),
        "China": (31.23, 121.47),
        "Brazil": (-23.55, -46.63),
        "Australia": (-33.86, 151.20)
    }

    region_names = list(regions.keys())
    np.random.seed(int(time.time()))

    points = []
    for _ in range(max(20, attack_count * 10)):
        r = np.random.choice(region_names)
        lat, lon = regions[r]

        points.append({
            "lat": lat + np.random.normal(0, 2),
            "lon": lon + np.random.normal(0, 2),
            "size": np.random.randint(50000, 150000)
        })

    points_df = pd.DataFrame(points)

    flows = []
    for _ in range(min(30, attack_count + 10)):
        src = np.random.choice(region_names)
        dst = np.random.choice(region_names)

        if src != dst:
            src_lat, src_lon = regions[src]
            dst_lat, dst_lon = regions[dst]

            flows.append({
                "from_lat": src_lat,
                "from_lon": src_lon,
                "to_lat": dst_lat,
                "to_lon": dst_lon
            })

    flow_df = pd.DataFrame(flows)

    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=points_df,
        get_position='[lon, lat]',
        get_radius="size",
        get_color='[0, 255, 255, 160]'
    )

    layer_lines = pdk.Layer(
        "ArcLayer",
        data=flow_df,
        get_source_position='[from_lon, from_lat]',
        get_target_position='[to_lon, to_lat]',
        get_source_color='[255, 0, 0, 160]',
        get_target_color='[0, 255, 255, 160]',
        get_width=5
    )

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2)

    st.pydeck_chart(pdk.Deck(
        layers=[layer_points, layer_lines],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10'
    ))

    st.markdown("---")

    # MODEL EVAL
    st.subheader("📊 Model Evaluation")
    y_pred_real, y_score_real = evaluate_real()

    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_real, y_pred_real)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        st.pyplot(fig)

    with col2:
        fpr, tpr, _ = roc_curve(y_real, y_score_real)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        st.pyplot(fig)

    st.markdown("---")

    # LOGS
    df = pd.DataFrame({
        "ID": idx,
        "Score": scores,
        "Prediction": ["ATTACK" if p==1 else "NORMAL" for p in pred]
    })

    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("Download Logs", df.to_csv(index=False), "logs.csv")

    if mode == "Real-Time":
        time.sleep(5)
        st.rerun()
