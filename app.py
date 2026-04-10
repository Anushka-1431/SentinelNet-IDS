# ==============================
# IMPORTS
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import joblib
import gdown
import time
import os
import pydeck as pdk

# 📩 EMAIL IMPORTS
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="SentinelNet IDS", layout="wide")

# ==============================
# UI STYLE
# ==============================
st.markdown("""
<style>
body { background-color: black; color: #00ffcc; font-family: monospace; }
h1 { text-shadow: 0 0 20px #00f7ff; }
.stMetric {
    background: #020617;
    border: 1px solid #00f7ff;
    padding: 15px;
    border-radius: 12px;
}
.stButton>button {
    background: black;
    border: 1px solid #00f7ff;
    color: #00f7ff;
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
# EMAIL ALERT FUNCTION
# ==============================
def send_email_alert(attack_count, attack_percent, severity, df):
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"
    receiver_email = "admin_email@gmail.com"

    subject = "🚨 SentinelNet ALERT: Intrusion Detected!"

    body = f"""
⚠️ ALERT: Network Intrusion Detected

Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Number of Attacks: {attack_count}
Attack Percentage: {attack_percent:.2f}%
Severity Level: {severity}

Top Suspicious Logs:
{df.head(5).to_string(index=False)}

Take immediate action!
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("✅ Email Sent")
    except Exception as e:
        print("❌ Email Failed:", e)

# ==============================
# LOADERS
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

if "last_email_time" not in st.session_state:
    st.session_state.last_email_time = 0

# ==============================
# HEADER
# ==============================
st.markdown("""
<h1 style='text-align:center;'>🚨 SentinelNet IDS</h1>
<p style='text-align:center;'>AI-Powered Intrusion Detection System</p>
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
    attack_percent = float(np.mean(pred) * 100)

    st.session_state.history.append(attack_percent)

    # ALERT SYSTEM
    if attack_count > 0:
        play_alert()

        if attack_percent > 50:
            severity_level = "HIGH"
        elif attack_percent > 20:
            severity_level = "MEDIUM"
        else:
            severity_level = "LOW"

        st.error(f"🚨 {attack_count} Intrusions Detected! | Severity: {severity_level}")

        current_time = time.time()

        if current_time - st.session_state.last_email_time > 60:
            if severity_level == "HIGH":
                df_email = pd.DataFrame({
                    "ID": idx,
                    "Score": scores,
                    "Prediction": ["ATTACK" if p==1 else "NORMAL" for p in pred]
                })

                send_email_alert(attack_count, attack_percent, severity_level, df_email)
                st.session_state.last_email_time = current_time
    else:
        st.success("✅ SYSTEM SECURE")

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("Traffic", len(pred))
    c2.metric("Attacks", attack_count)
    c3.metric("Attack %", f"{attack_percent:.2f}%")

    st.markdown("---")

    # PIE CHART
    st.subheader("🥧 Traffic Distribution")
    labels = ["Normal", "Attack"]
    sizes = [len(pred) - attack_count, attack_count]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig)

    st.markdown("---")

    # TREND
    st.subheader("📈 Attack Trend")
    st.line_chart(pd.DataFrame({"Attack %": st.session_state.history}))

    # NETWORK
    st.subheader("📡 Network Activity")
    st.line_chart(pd.DataFrame({"Traffic Score": scores}))

    st.markdown("---")

    # MODEL EVALUATION
    st.subheader("📊 Model Evaluation")

    y_pred_real, y_score_real = evaluate_real()

    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_real, y_pred_real)
        fig, ax = plt.subplots()
        ax.imshow(cm)

        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='white')

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
