# app_updated.py  —  Interface Friction Angle Prediction App (with uncertainty)
# Run:  streamlit run app_updated.py

import json
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="Interface Friction Angle Prediction", layout="wide")
st.title("Interface Friction Angle Prediction App")
st.markdown("### Predict φ (°) using a Random Forest model with uncertainty")

# --- Load model ---
# Expecting a scikit-learn RandomForestRegressor saved via joblib
model = joblib.load("phi_model_final.pkl")

# --- Sidebar inputs ---
st.sidebar.header("Input Parameters")

D10 = st.sidebar.text_input("D10 (mm)", value="0.25")
D30 = st.sidebar.text_input("D30 (mm)", value="0.35")
D60 = st.sidebar.text_input("D60 (mm)", value="0.55")

Dr = st.sidebar.slider("Relative Density Dr (%)", min_value=10, max_value=100, value=60, step=1)
ts = st.sidebar.slider("Tensile Strength (kN/m)", min_value=1, max_value=120, value=30, step=1)
aos = st.sidebar.text_input("Apparent Opening Size AOS (mm)", value="0.20")
elong = st.sidebar.slider("Elongation at Break (%)", min_value=5, max_value=100, value=20, step=1)
sigma = st.sidebar.slider("Normal Stress σₙ (kPa)", min_value=10, max_value=1000, value=100, step=5)

st.sidebar.header("Uncertainty")
conf_level = st.sidebar.selectbox("Confidence level", [90, 95], index=1)  # 95% default

st.markdown("---")

# --- Helper to check training coverage from optional JSON file ---
def check_training_coverage(values_dict):
    """
    values_dict = {"Dr": Dr, "ts": ts, "AOS": AOS_val, "elong": elong, "sigma": sigma, "Cu": Cu, "Cc": Cc}
    If 'training_ranges.json' exists in the working directory, it should look like:
    {
      "Dr": [10, 100],
      "ts": [1, 120],
      "AOS": [0.05, 1.0],
      "elong": [5, 100],
      "sigma": [10, 1000],
      "Cu": [1.2, 12.0],
      "Cc": [0.3, 4.0]
    }
    """
    try:
        with open("training_ranges.json", "r", encoding="utf-8") as f:
            rng = json.load(f)
    except Exception:
        return []  # no ranges file; no warnings

    out_of = []
    labels = {
        "Dr": "Dr (%)",
        "ts": "Tensile Strength (kN/m)",
        "AOS": "AOS (mm)",
        "elong": "Elongation at Break (%)",
        "sigma": "Normal Stress σₙ (kPa)",
        "Cu": "Cu (-)",
        "Cc": "Cc (-)",
    }
    for key, val in values_dict.items():
        if key not in rng:
            continue
        vmin, vmax = rng[key]
        if (val < vmin) or (val > vmax):
            # format numeric nicely
            def fmt(x):
                return f"{x:.4g}" if isinstance(x, (int, float)) else str(x)
            out_of.append(f"{labels.get(key, key)}: {fmt(val)} (train range {fmt(vmin)}–{fmt(vmax)})")
    return out_of

# --- Prediction button ---
if st.button("Predict"):
    try:
        # Parse numeric inputs (comma-safe)
        D10_val = float(D10.replace(",", "."))
        D30_val = float(D30.replace(",", "."))
        D60_val = float(D60.replace(",", "."))
        AOS_val = float(aos.replace(",", "."))

        # Derived gradation features
        Cu = D60_val / D10_val
        Cc = (D30_val ** 2) / (D10_val * D60_val)

        # Feature vector (match training order!)
        # [Dr, ts, AOS, elong, sigma, Cu, Cc]
        X_pred = np.array([[Dr, ts, AOS_val, elong, sigma, Cu, Cc]])

        # Point prediction
        phi_point = model.predict(X_pred)[0]

        # --- Prediction interval using tree-level distribution (RF) ---
        has_pi = False
        try:
            # For scikit RF, model.estimators_ is a list of fitted trees
            tree_preds = np.array([est.predict(X_pred)[0] for est in model.estimators_])
            alpha = 100 - conf_level
            lo = np.percentile(tree_preds, alpha / 2.0)
            hi = np.percentile(tree_preds, 100 - alpha / 2.0)
            has_pi = True
        except Exception:
            # If estimators_ not available (e.g., different model), skip PI silently
            has_pi = False

        # --- Display results ---
        col1, col2 = st.columns([1, 1])
        with col1:
            st.success(f"Predicted Interface Friction Angle φ: **{phi_point:.2f}°**")
            if has_pi:
                st.info(f"**{conf_level}% prediction interval:** [{lo:.2f}°, {hi:.2f}°]")

        with col2:
            st.markdown("#### Computed Parameters")
            st.write(f"Coefficient of Uniformity (Cu): **{Cu:.2f}**")
            st.write(f"Coefficient of Curvature (Cc): **{Cc:.2f}**")

        # --- Training coverage warnings (optional JSON) ---
        out_of_range = check_training_coverage(
            {"Dr": Dr, "ts": ts, "AOS": AOS_val, "elong": elong, "sigma": sigma, "Cu": Cu, "Cc": Cc}
        )
        if out_of_range:
            st.warning(
                "Some inputs are outside the model’s training range:\n\n- " + "\n- ".join(out_of_range)
            )

        # --- Low normal stress warning (per cross-campaign finding) ---
        if sigma <= 100:
            st.warning(
                "Low normal stress regime (≤100 kPa): cross-campaign tests showed larger errors in this range; "
                "interpret predictions with caution."
            )

    except Exception as e:
        st.error(f"Input error: {e}")

# --- Footer / small help ---
st.markdown("---")
st.caption(
    "Notes: Prediction intervals are derived from the distribution of tree-level predictions within the Random Forest ensemble. "
    "If a 'training_ranges.json' file is present, inputs are checked against training-domain ranges to surface coverage warnings."
)
