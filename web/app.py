
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import plotly.express as px
import plotly.graph_objects as go
import subprocess

# --- Setup Path & Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    from src.data import load_data, TabularDataset
    from src.modules import TabularDenoiseModel
    from src.diffusion import DSDDPM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- Configuration ---
st.set_page_config(
    page_title="DSDDPM Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("web/style.css")

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Sidebar ---
with st.sidebar:
    st.title("🔮 DSDDPM Studio")
    st.markdown("Dual-Scale Diffusion Probabilistic Model")
    st.markdown("---")
    page = st.radio("Navigation", ["Dashboard", "Data Explorer", "Train Model", "Generate Data", "Comparisons & Viz"])
    st.markdown("---")
    if not TORCH_AVAILABLE:
        st.error("PyTorch Missing")
    else:
        st.success("PyTorch Active")

# --- Page Logic ---
if page == "Dashboard":
    st.title("Dashboard")
    st.markdown("### Welcome to the Synthetic Data Studio")
    root = get_project_root()
    data_dir = os.path.join(root, "data")
    ckpt_dir = os.path.join(root, "checkpoints")
    datasets = [f for f in os.listdir(data_dir) if f.endswith(".csv")] if os.path.exists(data_dir) else []
    models = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Datasets", len(datasets))
    c2.metric("Models", len(models))
    c3.metric("Status", "Ready")
    
    st.markdown("### Quick Start")
    st.info("Start by uploading or selecting a dataset in the **Data Explorer**.")
    
    # Navigation Shortcuts
    st.markdown("### Quick Navigation")
    col1, col2, col3 = st.columns(3)
    if col1.button("📂 Go to Data Explorer"):
        # We can't switch radio programmatically easily in vanilla streamlit without session state tricks
        # But we can tell the user to click the sidebar, or use a query param
        st.experimental_set_query_params(page="Data Explorer")
        st.write("Please select 'Data Explorer' in the sidebar.")
    
    if col2.button("🚂 Go to Train Model"):
        st.write("Please select 'Train Model' in the sidebar.")
        
    if col3.button("✨ Go to Generate"):
        st.write("Please select 'Generate Data' in the sidebar.")
        
    # Visualization Placeholder

elif page == "Data Explorer":
    st.title("Data Explorer")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        # Save uploaded file
        root = get_project_root()
        data_dir = os.path.join(root, "data")
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name} to data/")
        
        df = pd.read_csv(save_path)
        st.dataframe(df.head())
    else:
        root = get_project_root()
        dummy = os.path.join(root, "data", "dummy.csv")
        if os.path.exists(dummy):
            df = pd.read_csv(dummy)
            st.dataframe(df.head())
            
            c1, c2 = st.columns(2)
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                c1.subheader(f"Numerical: {num_cols[0]}")
                c1.bar_chart(df[num_cols[0]].head(50))
                
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                c2.subheader(f"Categorical: {cat_cols[0]}")
                c2.bar_chart(df[cat_cols[0]].value_counts())

elif page == "Train Model":
    st.title("Train Diffusion Model")
    if not TORCH_AVAILABLE:
        st.error("PyTorch Required")
    else:
        root = get_project_root()
        data_dir = os.path.join(root, "data")
        datasets = [f for f in os.listdir(data_dir) if f.endswith(".csv")] if os.path.exists(data_dir) else []
        selected = st.selectbox("Select Dataset", datasets)
        epochs = st.number_input("Epochs", 1, 1000, 50)
        
        if st.button("Start Training"):
            path = os.path.join(data_dir, selected)
            cmd = [sys.executable, "-m", "src.train", "--data", path, "--epochs", str(epochs), "--batch_size", "64"]
            st.info(f"Running: {' '.join(cmd)}")
            
            with st.spinner("Training... check terminal/logs"):
                process = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    st.success("Training Complete!")
                    st.code(stdout)
                    
                    # Show download for latest model
                    ckpt_dir = os.path.join(root, "checkpoints")
                    latest_model = os.path.join(ckpt_dir, "model_final.pt")
                    if os.path.exists(latest_model):
                        with open(latest_model, "rb") as f:
                            st.download_button("Download Trained Model", f, file_name="model_final.pt")
                else:
                    st.error("Training Failed")
                    st.error(stderr)

elif page == "Generate Data":
    st.title("Generate Data")
    if not TORCH_AVAILABLE:
        st.error("PyTorch Required")
    else:
        root = get_project_root()
        ckpt_dir = os.path.join(root, "checkpoints")
        models = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
        data_dir = os.path.join(root, "data")
        datasets = [f for f in os.listdir(data_dir) if f.endswith(".csv")] if os.path.exists(data_dir) else []
        
        c1, c2 = st.columns(2)
        with c1:
            selected_model = st.selectbox("Model", models)
        with c2:
            base_data = st.selectbox("Base Schema", datasets)
            
        num = st.number_input("Samples", 10, 5000, 100)
        out_name = st.text_input("Output Name", "generated_data.csv")
        
        if st.button("Generate"):
            model_path = os.path.join(ckpt_dir, selected_model)
            data_path = os.path.join(data_dir, base_data)
            cmd = [sys.executable, "-m", "src.sample", "--model", model_path, "--data", data_path, "--num_samples", str(num), "--output", out_name]
            
            with st.spinner("Generating..."):
                res = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
                if res.returncode == 0:
                    st.success("Done!")
                    out_path = os.path.join(root, out_name)
                    if os.path.exists(out_path):
                        df = pd.read_csv(out_path)
                        st.dataframe(df.head())
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV", csv, "generated.csv", "text/csv")
                        
                        # Download Model Button
                        with open(model_path, "rb") as f:
                            st.download_button("Download Used Model (.pt)", f, file_name=selected_model)
                else:
                    st.error("Failed")
                    st.error(res.stderr)

elif page == "Comparisons & Viz":
    st.title("Visualization & Privacy Analysis")
    root = get_project_root()
    files = [f for f in os.listdir(root) if f.endswith(".csv")]
    sel = st.multiselect("Select Files (Select Real then Generated)", files, max_selections=2)
    
    if len(sel) > 0:
        dfs = []
        for f in sel:
            d = pd.read_csv(os.path.join(root, f))
            d['Source'] = f
            dfs.append(d)
        combined = pd.concat(dfs)
        
        st.subheader("Distribution Comparison")
        col = st.selectbox("Column", combined.columns)
        if combined[col].dtype == 'object':
             st.plotly_chart(px.histogram(combined, x=col, color='Source', barmode='group'))
        else:
             st.plotly_chart(px.histogram(combined, x=col, color='Source',  opacity=0.7, barmode='overlay'))

    # Privacy Analysis
    if len(sel) == 2:
        st.subheader("Privacy: Distance to Closest Record (DCR)")
        if st.checkbox("Run Privacy Check (DCR)"):
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import pairwise_distances
                
                real_name = sel[0]
                gen_name = sel[1]
                
                df_real = pd.read_csv(os.path.join(root, real_name)).select_dtypes(include=['number']).dropna()
                df_gen = pd.read_csv(os.path.join(root, gen_name)).select_dtypes(include=['number']).dropna()
                
                if df_real.empty or df_gen.empty:
                    st.warning("Needs numerical data for DCR check.")
                else:
                    # Normalize
                    scaler = StandardScaler()
                    scaler.fit(df_real)
                    
                    real_scaled = scaler.transform(df_real)
                    gen_scaled = scaler.transform(df_gen)
                    
                    # Compute distances
                    # For each gen point, find min dist to real
                    dists = pairwise_distances(gen_scaled, real_scaled)
                    min_dists = dists.min(axis=1)
                    
                    st.write(f"Mean DCR: {min_dists.mean():.4f}")
                    st.write(f"Min DCR: {min_dists.min():.4f} (Lower = closer to real data)")
                    
                    fig_dcr = px.histogram(min_dists, nbins=30, title="Distribution of Distances to Nearest Real Record")
                    fig_dcr.update_layout(xaxis_title="Distance", yaxis_title="Count")
                    st.plotly_chart(fig_dcr)
                    
                    if min_dists.min() < 0.01:
                        st.warning("⚠️ Some generated records are very close to real records. High privacy risk!")
                    else:
                        st.success("✅ Good separation from real records.")
                        
            except ImportError:
                st.error("Sklearn not available for privacy check.")
