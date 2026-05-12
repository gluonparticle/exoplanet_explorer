import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Exoplanet Explorer", layout="wide", page_icon="🪐", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    body, .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stButton>button { background: linear-gradient(90deg, #00E5FF 0%, #007BFF 100%); color: white; border-radius: 5px; border: none; font-weight: bold; width: 100%; }
    .metric-card { background-color: #161b22; padding: 20px; border-radius: 8px; border-left: 4px solid #00E5FF; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

st.markdown("<h1 style='color: #00E5FF;'>🪐 NASA TESS/Kepler Exoplanet Classifier</h1>", unsafe_allow_html=True)
st.write("Production Machine Learning Pipeline utilizing LightGBM Gradient Boosting.")
st.write("---")

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv("grand_dAAAAAAaataset_final_scaled.csv").head(100)
    except:
        return pd.DataFrame()

sample_df = load_sample_data()

st.subheader("🧪 Live Demo & Parameter Tuning")
st.write("Select a pre-processed telemetry frame from the TESS dataset. **You can manually double-click and edit the scaled physics parameters in the table** before running the inference.")

if not sample_df.empty:
    options = sample_df.index.tolist()
    selected_idx = st.selectbox("Select Test Subject (Row ID)", options, format_func=lambda x: f"Subject #{x} (Actual NASA Label: {sample_df.iloc[x]['koi_disposition']})")
    selected_data = sample_df.iloc[selected_idx].to_dict()
    actual_label = selected_data.pop('koi_disposition', 'Unknown')
else:
    st.warning("Sample dataset not found locally.")
    selected_data = {}
    actual_label = "Unknown"

col1, col2 = st.columns([1, 2])

with col2:
    st.write("**Editable Telemetry Features (Scaled)**")
    if selected_data:
        display_df = pd.DataFrame([selected_data]).T
        display_df.columns = ["Scaled Value"]
        
        # --- THE MAGIC SAUCE: Interactive Data Editor ---
        edited_df = st.data_editor(display_df, use_container_width=True, height=450)
        final_payload_data = edited_df["Scaled Value"].to_dict()
    else:
        final_payload_data = {}

with col1:
    st.write("### Execute Neural Inference")
    if st.button("🚀 Predict Disposition"):
        with st.spinner("Analyzing light curve telemetry..."):
            payload = {}
            for k, v in final_payload_data.items():
                clean_key = k.replace('LS+MCMC', 'LS_MCMC')
                payload[clean_key] = float(v)
            
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    prediction = response.json()['prediction']
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>AI Classification Output:</h3>
                        <h1 style="color: #00FF7F;">{prediction}</h1>
                        <p>Actual NASA Label: <b>{actual_label}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Backend Connection Failed: {e}")

st.write("---")
with st.expander("📁 Batch Inference (CSV Upload)"):
    uploaded_file = st.file_uploader("Upload CSV file with normalized columns", type=["csv"])
    if uploaded_file and st.button("Run Batch Prediction"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{API_URL}/predict_csv", files=files)
            if response.status_code == 200:
                st.success("Batch Prediction complete!")
                st.download_button("Download Results CSV", response.content, "predictions.csv", "text/csv")
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
