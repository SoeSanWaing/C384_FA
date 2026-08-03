# Smart Hawker Centre Streamlit App

The app loads the selected 50-epoch YOLO11n checkpoint from:

`models_yolo11n_hp/run1_baseline_50epochs_best.pt`

It supports image upload, a browser camera snapshot, and uploaded video. Pest and
cleaning alerts are evaluated independently.

## Run in the existing `c384` environment

Open **Anaconda Prompt** and run:

```powershell
conda activate c384
cd C:\Users\c384\Documents\FA\FA_Final
streamlit run streamlit_app.py
```

If the environment does not contain the required packages, install them with:

```powershell
pip install -r requirements_streamlit.txt
```

The local dashboard will normally open at `http://localhost:8501`.

## Scope

This is a prototype dashboard. Its alerts appear inside Streamlit only. A production
deployment would connect the alert states to an authenticated notification service and
would use longer camera-specific persistence rules.
