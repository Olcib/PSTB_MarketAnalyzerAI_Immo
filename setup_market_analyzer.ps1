param(
    [string]$ProjectName = "MarketAnalyzerAI-Immo",
    [switch]$OpenVSCode = $true
)

function Write-Utf8($Path, $Content) {
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $Content | Out-File -FilePath $Path -Encoding utf8 -Force
}

$root = Join-Path -Path (Get-Location) -ChildPath $ProjectName
New-Item -ItemType Directory -Path $root -Force | Out-Null

$folders = @(
    "data","models","vectorstore","notebooks","src",
    "streamlit_app","streamlit_app/pages","streamlit_app/assets",".vscode"
)
foreach ($f in $folders) { New-Item -ItemType Directory -Path (Join-Path $root $f) -Force | Out-Null }

$gitignore = @'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.ipynb_checkpoints/
.venv/
venv/
dist/
build/

# Secrets
.env

# Data & artifacts
data/*
!data/.gitkeep
models/*
!models/.gitkeep
vectorstore/*
!vectorstore/.gitkeep

# VS Code
.vscode/
'@
Write-Utf8 (Join-Path $root ".gitignore") $gitignore
Write-Utf8 (Join-Path $root "data/.gitkeep") ""
Write-Utf8 (Join-Path $root "models/.gitkeep") ""
Write-Utf8 (Join-Path $root "vectorstore/.gitkeep") ""

$requirements = @'
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
tensorflow
torch
transformers
langchain
chromadb
faiss-cpu
openai
streamlit
plotly
altair
python-dotenv
'@
Write-Utf8 (Join-Path $root "requirements.txt") $requirements

$envExample = @'
# Rename to .env and fill your keys
OPENAI_API_KEY="your_openai_api_key"
HUGGINGFACEHUB_API_TOKEN="your_hf_token"
'@
Write-Utf8 (Join-Path $root ".env.example") $envExample

$readme = @'
# MarketAnalyzerAI Immo — Strategic AI Co-Pilot for Real-Estate Rentals

End-to-end AI: competitive analysis, forecasting, tailored recommendations, RAG chatbot, and a "What-if?" revenue simulator.
Replace this README with your full version when ready.
'@
Write-Utf8 (Join-Path $root "README.md") $readme

$vscodeSettings = @'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true
}
'@
Write-Utf8 (Join-Path $root ".vscode/settings.json") $vscodeSettings

$nbSkeleton = @'
{
  "cells": [],
  "metadata": { "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" } },
  "nbformat": 4,
  "nbformat_minor": 5
}
'@
$nbFiles = @("eda.ipynb","ml_baseline.ipynb","ml_advanced.ipynb","dl_experiments.ipynb","nlp_transformers.ipynb")
foreach ($n in $nbFiles) { Write-Utf8 (Join-Path $root ("notebooks/" + $n)) $nbSkeleton }

$data_utils = @'
from pathlib import Path
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_csv(p)

def save_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
'@
Write-Utf8 (Join-Path $root "src/data_utils.py") $data_utils

$train_dl = @'
def main():
    # TODO: implement MLP/LSTM training (Keras/PyTorch)
    print("Stub DL training script ready.")

if __name__ == "__main__":
    main()
'@
Write-Utf8 (Join-Path $root "src/train_dl.py") $train_dl

$app = @'
import streamlit as st

st.set_page_config(page_title="MarketAnalyzerAI Immo", layout="wide")
st.title("MarketAnalyzerAI Immo — Strategic AI Co-Pilot")

tabs = st.tabs(["EDA", "Forecasts", "Chatbot (RAG)", "What-if Simulator"])

with tabs[0]:
    st.subheader("Exploratory Data Analysis")
    st.info("Add plots and dataset summaries here.")

with tabs[1]:
    st.subheader("ML/DL Forecasts")
    st.info("Show model predictions and KPIs (occupancy, revenue).")

with tabs[2]:
    st.subheader("RAG Chatbot")
    st.info("Ask domain questions grounded in your documents.")

with tabs[3]:
    st.subheader("What-if Simulator")
    price = st.slider("Price (€/night)", 30, 300, 120, 5)
    promo = st.slider("Promotion (%)", 0, 50, 10, 1)
    amenities = st.multiselect("Added amenities", ["A/C", "Washer", "Dryer", "Dishwasher", "High-speed WiFi"])
    st.success(f"Simulated: price={price}, promo={promo}%, amenities={amenities}")
'@
Write-Utf8 (Join-Path $root "streamlit_app/app.py") $app

try {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv (Join-Path $root ".venv") | Out-Null
} catch {
    Write-Warning "Could not create venv automatically. Create it manually if needed."
}

if ($OpenVSCode) {
    try {
        Set-Location $root
        code .
    } catch {
        Write-Warning "VS Code 'code' command not found. Open manually: $root"
    }
}

Write-Host "✅ Project scaffold created at: $root" -ForegroundColor Green
