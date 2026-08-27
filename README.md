# Nigerian Traffic Crashes — Simple Project Guide

This project predicts `Total_Crashes` (how many crashes happened) for
Nigerian states, using recorded contributing factors. Everything is written
in plain, simple terms and split into just a few files.

## Files in This Project

```
traffic-crash-project-simple/
├── Nigerian_Road_Traffic_Crashes_2020_2024.csv        <- the raw data
├── notebooks/
│   └── traffic_crashes_analysis.ipynb                <- full walkthrough with charts
├── saved_model/                                       <- created after you run train.py
├── data_helper.py      <- shared cleaning steps (used everywhere)
├── train.py             <- teaches the models and saves the best one
├── fastapi_app.py        <- ONE file: the whole web API
├── streamlit_app.py      <- ONE file: the whole interactive website
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Step-by-Step: How to Run Everything

### Step 1 — Teach the model (do this first!)
```bash
python train.py
```
This reads the data, cleans it, teaches two different models, picks the one
that makes smaller mistakes, and saves it inside a folder called
`saved_model/`. You only need to do this once (or again later if you change
the data).

### Step 2 — Look through the notebook
```bash
jupyter notebook notebooks/traffic_crashes_analysis.ipynb
```
This shows every step in detail: loading the data, cleaning it, charts with
plain-English explanations, teaching the models, and what the results mean.

### Step 3 — Start the web API
```bash
uvicorn fastapi_app:app --reload
```
Then open `http://127.0.0.1:8000/docs` in your browser to try it out, or
send a request from the terminal:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"State":"Lagos","Year":2023,"Quarter_Num":2,"SPV":20,"DAD":5,"PWR":2,"FTQ":1,"Other_Factors":10}'
```

### Step 4 — Start the interactive website
```bash
streamlit run streamlit_app.py
```
Use the green menu on the left to move through: Home → Load Data → Clean
Data & Charts → Train Models → Make a Prediction.

## A Few Simple Terms Explained

- **joblib** — just a tool for saving a trained model to a file (like
  saving a Word document), so we can reload it later without re-teaching it
  from scratch every time.
- **saved_model folder** — a normal folder on your computer where the
  trained model and its helper files are kept.
- **DataPreparer** — our own small tool that turns raw data (like state
  names) into the number format the model needs. It "learns" once from the
  training data, then reuses what it learned every time after that.
- **Target** — the number we're trying to predict (`Total_Crashes`).
- **Features / clues** — the pieces of information the model looks at to
  make its guess (state, quarter, speed violations, etc.).
- **Training / Validation / Test data** — three separate slices of the
  data: one to teach the model, one to compare models and pick a winner,
  and one kept completely aside for one final, honest check at the end.
