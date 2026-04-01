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

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="SentinelNet IDS", layout="wide")

# ==============================
# GOOGLE DRIVE DOWNLOADER (FIXED)
# ==============================
def download_file_from_drive(file_id, destination):
    if os.path.exists(destination):
        return

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

import gdown
import os

def load_pkl_from_drive(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return joblib.load(filename)


def load_csv_from_drive(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
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
# REAL EVAL
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
        st.error(f"🚨 {attack_count} Intrusions Detected!")
    else:
        st.success("✅ System Secure")

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("Traffic", len(pred))
    c2.metric("Attacks", attack_count)
    c3.metric("Attack %", f"{attack_percent:.2f}%")

    st.markdown("---")

    # TREND
    st.subheader("📈 Attack Trend")
    st.line_chart(pd.DataFrame({"Attack %": st.session_state.history}))

    st.markdown("---")

    # ==============================
    # MODEL EVALUATION
    # ==============================
    st.subheader("📊 Model Evaluation")

    with st.spinner("🔍 Evaluating model..."):
        y_pred_real, y_score_real = evaluate_real()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_real, y_pred_real):.2f}")
    c2.metric("Precision", f"{precision_score(y_real, y_pred_real):.2f}")
    c3.metric("Recall", f"{recall_score(y_real, y_pred_real):.2f}")
    c4.metric("F1", f"{f1_score(y_real, y_pred_real):.2f}")

    # ==============================
    # SIDE BY SIDE FIXED
    # ==============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_real, y_pred_real)

        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(cm)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha='center', va='center')

        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(y_real, y_score_real)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.plot([0,1],[0,1],'--')
        ax.legend()

        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # ==============================
    # DONUT
    # ==============================
    st.subheader("📊 Distribution")

    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(
        [normal_count, attack_count],
        autopct='%1.1f%%',
        colors=["#00f7ff", "#ff4d4d"],
        wedgeprops={'width':0.35}
    )
    st.pyplot(fig)

    st.write(f"🔵 Normal: {normal_count}")
    st.write(f"🔴 Attack: {attack_count}")

    st.markdown("---")

    # ==============================
    # LOGS
    # ==============================
    df = pd.DataFrame({
        "ID": idx,
        "Score": scores,
        "Prediction": ["ATTACK" if p==1 else "NORMAL" for p in pred]
    })

    st.dataframe(df.head(30))
    st.download_button("Download Logs", df.to_csv(index=False), "logs.csv")

    # AUTO REFRESH
    if mode == "Real-Time":
        time.sleep(5)
        st.rerun()
