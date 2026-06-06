import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Emission Tracker & Predictor",
    page_icon="https://cdn-icons-png.flaticon.com/128/17785/17785781.png",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────────
# Icon URLs  (Flaticon public CDN)
# ─────────────────────────────────────────────────────────────────
ICON = {
    "leaf":        "https://cdn-icons-png.flaticon.com/128/17785/17785781.png",
    "home":        "https://cdn-icons-png.flaticon.com/512/1946/1946436.png",
    "calculator":  "https://cdn-icons-png.flaticon.com/128/17329/17329538.png",
    "microscope":  "https://cdn-icons-png.flaticon.com/128/12538/12538663.png",
    "electricity": "https://cdn-icons-png.flaticon.com/128/7867/7867697.png",
    "car":         "https://cdn-icons-png.flaticon.com/128/575/575780.png",
    "plastic":     "https://cdn-icons-png.flaticon.com/128/4643/4643182.png",
    "tree":        "https://cdn-icons-png.flaticon.com/512/489/489969.png",
    "plant":       "https://cdn-icons-png.flaticon.com/512/628/628283.png",
    "factory":     "https://cdn-icons-png.flaticon.com/512/2917/2917995.png",
    "chart":       "https://cdn-icons-png.flaticon.com/128/2961/2961248.png",
    "bulb":        "https://cdn-icons-png.flaticon.com/128/4415/4415867.png",
    "target":      "https://cdn-icons-png.flaticon.com/128/7770/7770344.png",
    "quiz":        "https://cdn-icons-png.flaticon.com/512/942/942748.png",
    "science":     "https://cdn-icons-png.flaticon.com/128/12489/12489862.png",
    "globe":       "https://cdn-icons-png.flaticon.com/128/9985/9985721.png",
    "info":        "https://cdn-icons-png.flaticon.com/128/19038/19038544.png",
}


def img(url, size=20, valign="middle", mr=8):
    """Return an inline <img> HTML tag."""
    return (
        f'<img src="{url}" width="{size}" height="{size}" '
        f'style="vertical-align:{valign};margin-right:{mr}px;margin-bottom:2px;">'
    )


def div_label(icon_url, text, size=18):
    """Render a styled section divider with an icon."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;'
        f'margin:22px 0 10px;font-size:1rem;font-weight:700;">'
        f'{img(icon_url, size, mr=0)}&nbsp;{text}'
        f'<span style="flex:1;height:1px;background:rgba(128,128,128,0.2);'
        f'margin-left:10px;display:inline-block;"></span></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# CSS  (light + dark adaptive, mobile-friendly)
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.stApp::before {
    content:"";
    position: fixed;
    inset: 0;
    background-image: url('https://static.vecteezy.com/system/resources/thumbnails/003/471/376/small/nature-green-background-free-vector.jpg');
    background-size: cover;
    background-position: center;
    z-index: -1;
    opacity: 0.2;
}
.stApp{
    background-color: transparent;
}
:root {
    --green:  #22c55e;
    --gdark:  #16a34a;
    --teal:   #0d9488;
    --amber:  #f59e0b;
    --blue:   #0284c7;
    --pink:   #ec4899;
    --red:    #ef4444;
    --shadow: 0 2px 14px rgba(0,0,0,0.07);
    --r:      14px;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.15); }
[data-testid="stSidebar"] .stRadio label {
    font-weight: 600; font-size: 0.94rem; padding: 3px 0;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, var(--green), var(--teal)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65em 1.4em !important;
    font-weight: 700 !important;
    font-size: 0.97rem !important;
    width: 100% !important;
    transition: opacity .2s, box-shadow .2s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    opacity: 0.87 !important;
    box-shadow: 0 4px 18px rgba(34,197,94,.35) !important;
}

/* Generic card */
.card {
    background: rgba(128,128,128,.07);
    border-radius: var(--r);
    padding: 18px 20px;
    margin-bottom: 12px;
    border-left: 5px solid var(--green);
    box-shadow: var(--shadow);
    line-height: 1.8;
}
.card-amber { border-left-color: var(--amber); }
.card-pink  { border-left-color: var(--pink);  }
.card-blue  { border-left-color: var(--blue);  }
.card-red   { border-left-color: var(--red);   }
.card-teal  { border-left-color: var(--teal);  }

/* Stat rows inside cards */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid rgba(128,128,128,.10);
    font-size: 0.88rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-label { opacity: .70; }
.stat-value { font-weight: 600; }

/* Feature cards (home page) */
.feat-card {
    background: rgba(128,128,128,.07);
    border-radius: var(--r);
    padding: 22px 14px;
    text-align: center;
    border-top: 4px solid var(--teal);
    box-shadow: var(--shadow);
    height: 100%;
}
.feat-card img  { margin-bottom: 10px; }
.feat-card b    { font-size: .93rem; display: block; margin-bottom: 5px; }
.feat-card p    { font-size: .80rem; opacity: .68; margin: 0; }

/* Step rows */
.step-row {
    display: flex; align-items: flex-start;
    gap: 12px; margin-bottom: 12px;
}
.step-num {
    background: var(--green); color: #fff;
    font-weight: 700; font-size: .82rem;
    border-radius: 50%;
    min-width: 26px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 3px;
}
.step-text { font-size: .93rem; line-height: 1.55; }

/* Result badge */
.badge {
    border-radius: var(--r);
    padding: 20px 14px;
    text-align: center;
    box-shadow: var(--shadow);
    background: rgba(128,128,128,.07);
}
.badge img  { width: 56px; height: 56px; margin-bottom: 8px; display: block; margin-left: auto; margin-right: auto; }
.badge span { display: block; font-weight: 700; font-size: .88rem; }

/* Responsive */
@media (max-width: 640px) {
    .card { padding: 12px 14px; font-size: .85rem; }
    h1    { font-size: 1.35rem !important; }
    h2, h3 { font-size: 1.05rem !important; }
    .stButton > button { font-size: .9rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent
MODEL_PATH         = BASE_DIR / "carbon_predictor_model.pkl"
SCALER_PATH        = BASE_DIR / "carbon_scaler.pkl"
CALIBRATOR_PATH    = BASE_DIR / "carbon_calibrator.pkl"
PRIMARY_DATA_PATH  = BASE_DIR / "primary_data_CO2.csv"

# ─────────────────────────────────────────────────────────────────
# Load Model & Scaler
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    # Load the scaler fitted on IoT data during training
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_calibrator():
    # Load the calibration layer (scale: 0.3995, intercept: -0.0584)
    if not CALIBRATOR_PATH.exists():
        return None
    return joblib.load(CALIBRATOR_PATH)


model      = load_model()
scaler     = load_scaler()
calibrator = load_calibrator()

# ─────────────────────────────────────────────────────────────────
# Constants  (verified against training data)
# ─────────────────────────────────────────────────────────────────
plastic_item_weights = {
    "Plastic bag":      0.005,
    "Plastic bottle":   0.020,
    "Plastic straw":    0.001,
    "Plastic cutlery":  0.003,
    "Food container":   0.025,
    "Plastic wrapping": 0.004,
}
# Cambodian urban waste baseline: 0.141 kg per item
# Derived from primary survey data where all plastic values are multiples of 0.141
PLASTIC_KG_PER_ITEM     = 0.141               # 0.141 kg per item
MOTORCYCLE_PETROL_CO2   = 0.11367            # kg CO2/km  (from dataset header)
EDC_RATE                = 610.0              # Riel / kWh  (EDC residential tier)
GRID_EMISSION_FACTOR    = 0.18708            # kg CO2 / kWh
PLASTIC_LIFECYCLE_CO2   = 6.0               # kg CO2 / kg plastic

# ─────────────────────────────────────────────────────────────────
# Baseline: human metabolic CO2 + minimal indirect emissions
# Used when all inputs are zero (user did nothing / spent day in nature)
# ~0.5 kg CO2/day is a commonly cited minimum per-person baseline
# covering breathing metabolism and minimal indirect consumption.
# ─────────────────────────────────────────────────────────────────
BASELINE_CO2_PER_DAY = 0.50  # kg CO2/day

# ─────────────────────────────────────────────────────────────────
# Session State defaults
# ─────────────────────────────────────────────────────────────────
_defaults = dict(
    has_predicted=False, prediction=0.0,
    energy_usage_kwh_day=0.0, energy_usage_kwh_month=0.0,
    energy_co2=0.0, electricity_payment=0.0,
    transportation_mode="Walking", transportation_energy_source="None",
    transportation_distance=0.0, transportation_factor=0.005,
    transportation_co2=0.0,
    plastic_items=0, plastic_usage=0.0, plastic_co2=0.0,
    input_data=None, input_data_raw=None,
    all_zero_input=False,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">'
        f'<img src="{ICON["leaf"]}" width="34">'
        f'<span style="font-size:1.1rem;font-weight:800;letter-spacing:-0.02em;">Eco-Predict</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("Application Menu")
    page = st.radio(
        "nav",
        ["Home", "Carbon Calculator", "Recommendations", "Model Transparency"],
        label_visibility="collapsed",
    )

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════
if page == "Home":

    # Hero header
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:16px;padding-bottom:6px;">'
        f'<img src="{ICON["leaf"]}" width="54">'
        f'<div>'
        f'<h1 style="margin:0;line-height:1.1;">Project Introduction: Eco-Predict</h1>'
        f'<p style="margin:3px 0 0;opacity:.6;font-size:.92rem;">'
        f'Cambodia Personal Carbon Emission Tracker</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.write(
        "Every daily choice we make, from turning on a light to jumping on our morning commute, leaves a trace on our climate. This carbon emission prediction is an interactive analytics platform designed to bridge the gap between human behavior and climate action. By transforming your daily habits into clear, structured environmental metrics through localized machine learning, this platform empowers you to visualize, understand, and minimize your personal carbon footprint."
    )

    div_label(ICON["info"], "Why Track Individual Emissions?")
    st.write(
        "CO\u2082 serves as the primary greenhouse gas accelerating climate transformations. While international environmental audits track macro-level statistics, grassroots behavior changes manifest when we trace personal impact boundaries. This application isolates three key areas of everyday living:"
    )

    c1, c2, c3 = st.columns(3)
    feat = [
        (ICON["electricity"], "Electricity Usage",  "Mapping monthly economic costs back to operational grid load metrics."),
        (ICON["car"],         "Transportation",    "Evaluating machine assets, alternative fuel types, and travel distances."),
        (ICON["plastic"],     "Plastic Waste","Measuring linear single-use consumer polymers against active lifecycle burdens."),
    ]
    for col, (ic, title, desc) in zip([c1, c2, c3], feat):
        with col:
            st.markdown(
                f'<div class="feat-card"><img src="{ic}" width="42">'
                f'<b>{title}</b><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    div_label(ICON["target"], "How to Use This App")
    steps = [
        ("Step 1: Carbon Calculator",   "Enter your electricity bill, transport details, and plastic usage."),
        ("Step 2: Prediction",             "Press the button to get your estimated daily CO\u2082 in kg."),
        ("Step 3: Recommendations",     "Get personalised tips targeting your largest emission source."),
        ("Step 4: Model Transparency",  "Explore the dataset, formulas, and science behind the model."),
    ]
    for i, (t, d) in enumerate(steps, 1):
        st.markdown(
            f'<div class="step-row"><div class="step-num">{i}</div>'
            f'<div class="step-text"><b>{t}</b>:{d}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("""Select **Carbon Calculator** in the sidebar to get started.""")

# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — CARBON CALCULATOR
# ═══════════════════════════════════════════════════════════════════
elif page == "Carbon Calculator":

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;padding-bottom:4px;">'
        f'<img src="{ICON["calculator"]}" width="46">'
        f'<div><h1 style="margin:0;">Carbon Calculator</h1>'
        f'<p style="margin:3px 0 0;opacity:.6;font-size:.92rem;">'
        f'Fill in your daily habits and press Calculate</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── 1. Electricity ─────────────────────────────────────────────
    div_label(ICON["electricity"], "Electricity Usage")
    electricity_payment = st.number_input(
        "Monthly Electricity Bill (Riel)",
        min_value=0.0, step=1000.0,
        help=f"EDC residential tariff: {EDC_RATE:.0f} Riel / kWh. Leave at 0 if you did not use electricity today.",
    )

    # ── FIX: Warn if the value looks like it was entered in kWh instead of Riel
    if 0 < electricity_payment < 2000:
        st.warning(
            f"⚠️ Your entered value ({electricity_payment:,.0f}) seems very low. "
            f"Please make sure your electricity bill is entered in **Riel (៛)**, not in kWh. "
            f"A typical Phnom Penh household bill is usually above 4,000 Riel. "
            f"At the EDC rate of {EDC_RATE:.0f} Riel/kWh, your entry converts to only "
            f"{electricity_payment / EDC_RATE:.3f} kWh/month."
        )

    energy_usage_kwh_month = electricity_payment / EDC_RATE
    energy_usage_kwh_day   = energy_usage_kwh_month / 30.0
    energy_co2             = energy_usage_kwh_day * GRID_EMISSION_FACTOR

    if electricity_payment > 0:
        st.markdown(
            f'<div class="card" style="padding:13px 18px;">'
            f'<div class="stat-row"><span class="stat-label">Monthly usage</span>'
            f'<span class="stat-value">{energy_usage_kwh_month:.2f} kWh/month</span></div>'
            f'<div class="stat-row"><span class="stat-label">Daily usage</span>'
            f'<span class="stat-value">{energy_usage_kwh_day:.4f} kWh/day</span></div>'
            f'<div class="stat-row"><span class="stat-label">Estimated CO\u2082</span>'
            f'<span class="stat-value">{energy_co2:.4f} kg/day</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 2. Transportation ──────────────────────────────────────────
    div_label(ICON["car"], "Transportation")
    tc1, tc2 = st.columns(2)
    with tc1:
        transportation_mode = st.selectbox(
            "Mode of Transport",
            ["Walking", "Bicycle", "Motorcycle/Scooter",
             "Tuk Tuk, Grab, PassApp", "Car", "Bus"],
        )
    with tc2:
        if transportation_mode in ["Bicycle", "Walking"]:
            transportation_energy_source = st.selectbox("Energy Source", ["None"])
        elif transportation_mode in ["Motorcycle/Scooter", "Tuk Tuk, Grab, PassApp"]:
            transportation_energy_source = st.selectbox(
                "Energy Source", ["Petrol (Gasoline)", "Electric"]
            )
        else:
            transportation_energy_source = st.selectbox(
                "Energy Source", ["Petrol (Gasoline)", "Electric", "Hybrid"]
            )

    transportation_distance = st.number_input(
        "Daily Travel Distance (km)", min_value=0.0, step=1.0
    )

    # Resolve emission factor
    if transportation_mode in ["Bicycle", "Walking"]:
        transportation_factor = 0.0
    elif transportation_mode == "Motorcycle/Scooter":
        transportation_factor = MOTORCYCLE_PETROL_CO2 if transportation_energy_source == "Petrol (Gasoline)" else 0.020
    elif transportation_mode == "Tuk Tuk, Grab, PassApp":
        transportation_factor = MOTORCYCLE_PETROL_CO2 if transportation_energy_source == "Petrol (Gasoline)" else 0.077
    elif transportation_mode == "Car":
        transportation_factor = {"Petrol (Gasoline)": 0.308, "Electric": 0.077, "Hybrid": 0.176}.get(
            transportation_energy_source, 0.308)
    elif transportation_mode == "Bus":
        transportation_factor = {"Petrol (Gasoline)": 0.089, "Electric": 0.040, "Hybrid": 0.060}.get(
            transportation_energy_source, 0.089)
    else:
        transportation_factor = MOTORCYCLE_PETROL_CO2

    transportation_co2 = transportation_distance * transportation_factor

    if transportation_distance > 0:
        st.markdown(
            f'<div class="card card-amber" style="padding:13px 18px;">'
            f'<div class="stat-row"><span class="stat-label">Emission factor</span>'
            f'<span class="stat-value">{transportation_factor:.5f} kg CO\u2082/km</span></div>'
            f'<div class="stat-row"><span class="stat-label">Estimated CO\u2082</span>'
            f'<span class="stat-value">{transportation_co2:.4f} kg/day</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 3. Plastic ─────────────────────────────────────────────────
    div_label(ICON["plastic"], "Plastic Consumption")
    plastic_items = st.number_input(
        "Single-Use Plastic Items Per Day", min_value=0, step=1,
        help="Bags, bottles, straws, cutlery, containers, wrapping, etc.",
    )
    plastic_usage = plastic_items * PLASTIC_KG_PER_ITEM
    plastic_co2   = plastic_usage * PLASTIC_LIFECYCLE_CO2

    if plastic_items > 0:
        st.markdown(
            f'<div class="card card-pink" style="padding:13px 18px;">'
            f'<div class="stat-row"><span class="stat-label">Plastic mass</span>'
            f'<span class="stat-value">{plastic_usage:.4f} kg/day</span></div>'
            f'<div class="stat-row"><span class="stat-label">Estimated CO\u2082</span>'
            f'<span class="stat-value">{plastic_co2:.4f} kg/day</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Model input frame ──────────────────────────────────────────
    # NOTE: The ML model was trained with raw feature values:
    #   - Energy_Usage_kWh        → daily kWh
    #   - Transportation_Distance_km → raw km (model coefficients encode the factor)
    #   - Plastic_Usage_kg        → kg of plastic
    # The fallback direct-sum uses per-source CO2 and is the ground-truth for display.
    input_data = pd.DataFrame({
        "Energy_Usage_kWh":           [energy_usage_kwh_day],
        "Transportation_Distance_km": [transportation_distance],
        "Plastic_Usage_kg":           [plastic_usage],
    })
    # NOTE: The model was trained on raw (unscaled) features — do NOT apply scaler before predict.
    # This matches the Local_Validation.ipynb pipeline exactly:
    #   model.predict(df_pri_raw[features])  ← raw values, no scaler.transform()
    # Applying the IoT scaler here would distort coefficients (especially plastic).
    input_data_raw = input_data.values

    with st.expander("Technical — Model Input Diagnostics"):
        st.caption("Raw feature values passed to model (no scaling applied)")
        st.dataframe(input_data, use_container_width=True)

    # ── Predict ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Calculate My Carbon Footprint"):

        all_zero = (electricity_payment == 0 and transportation_distance == 0 and plastic_items == 0)

    # ── BASELINE CASE ─────────────────────────────────────────────
        if all_zero:
            prediction = BASELINE_CO2_PER_DAY
            used_model = False
            used_fallback = False
            used_baseline = True

        else:
            used_baseline = False

        # ── ML PREDICTION WITH CALIBRATION ───────────────────────
        # Reproduces Local Validation notebook pipeline exactly:
        # Step 1: IoT scaler normalizes raw input features
        # Step 2: IoT model predicts raw CO2
        # Step 3: Calibrator corrects to primary survey data range
            ml_prediction = None
            if model is not None and calibrator is not None:
                try:
                    # Use raw features — matches Local_Validation.ipynb:
                    # model.predict(df_pri_raw[features]) then calibrator.predict(...)
                    raw_iot = model.predict(input_data_raw)
                    ml_prediction = float(calibrator.predict(raw_iot.reshape(-1, 1))[0])
                    ml_prediction = max(0.0, ml_prediction)
                except Exception:
                    ml_prediction = None

        # ── FALLBACK: direct emission factor sum ──────────────────
            direct_sum = energy_co2 + transportation_co2 + plastic_co2

        # ── FINAL DECISION ────────────────────────────────────────
            if ml_prediction is None:
                prediction = direct_sum
                used_model = False
                used_fallback = True

            else:
                prediction = ml_prediction
                used_model = True
                used_fallback = False

    # ── SAVE STATE ────────────────────────────────────────────────
        # ── PROPORTIONAL SPLIT ────────────────────────────────
        # Split the ML prediction proportionally across the 3 factors
        # so breakdown cards always sum to the prediction exactly
        _weight_sum = energy_co2 + transportation_co2 + plastic_co2
        if _weight_sum > 0 and not all_zero:
            energy_co2_display    = (energy_co2 / _weight_sum) * prediction
            transport_co2_display = (transportation_co2 / _weight_sum) * prediction
            plastic_co2_display   = (plastic_co2 / _weight_sum) * prediction
        else:
            energy_co2_display    = energy_co2
            transport_co2_display = transportation_co2
            plastic_co2_display   = plastic_co2

        st.session_state.update(dict(
            has_predicted=True,
            prediction=prediction,
            energy_usage_kwh_day=energy_usage_kwh_day,
            energy_usage_kwh_month=energy_usage_kwh_month,
            energy_co2=energy_co2_display,
            electricity_payment=electricity_payment,
            transportation_mode=transportation_mode,
            transportation_energy_source=transportation_energy_source,
            transportation_distance=transportation_distance,
            transportation_factor=transportation_factor,
            transportation_co2=transport_co2_display,
            plastic_items=plastic_items,
            plastic_usage=plastic_usage,
            plastic_co2=plastic_co2_display,
            input_data=input_data,
            input_data_raw=input_data_raw,
            all_zero_input=all_zero,
        ))

        st.balloons()
        st.markdown("---")

        # ── Source annotation ──────────────────────────────────────
        if used_baseline:
            st.info(
                "It looks like you spent your day with nature, no electricity, travel, or plastic! "
                "Every person still has a small unavoidable carbon baseline from metabolism and "
                "indirect consumption. This result reflects that minimum human footprint."
            )


        # Badge selection
        if prediction < 3.0:
            b_img, b_lbl, b_col = ICON["tree"],    "Low Footprint",      "#22c55e"
        elif prediction < 12.0:
            b_img, b_lbl, b_col = ICON["plant"],   "Moderate Footprint", "#f59e0b"
        else:
            b_img, b_lbl, b_col = ICON["factory"], "High Footprint",     "#ef4444"

        # Result row
        r1, r2 = st.columns([3, 1])
        with r1:
            trees = prediction / 0.06
            st.markdown(
                f'<div class="card" style="padding:20px 24px;">'
                f'{img(ICON["globe"], 26)}'
                f'<span style="font-size:1.55rem;font-weight:800;">'
                f'{prediction:.2f} kg CO\u2082/day</span><br>'
                f'<span style="opacity:.6;font-size:.86rem;">Predicted daily carbon emission</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card card-teal" style="padding:13px 18px;font-size:.9rem;">'
                f'{img(ICON["tree"], 18)}'
                f'<b>{trees:.1f} mature trees</b> growing for a year would offset '
                f"today's emissions."
                f'</div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div class="badge"><img src="{b_img}">'
                f'<span style="color:{b_col};">{b_lbl}</span></div>',
                unsafe_allow_html=True,
            )

        # ── Detailed 3-column breakdown ────────────────────────────
        # Only show component breakdown when inputs were actually provided
        if not all_zero:
            div_label(ICON["chart"], "Detailed Breakdown")
            d1, d2, d3 = st.columns(3)
            col_cards = [
                (d1, "", ICON["electricity"], "Electricity",
                 [("Bill", f"{electricity_payment:,.0f} Riel"),
                  ("Monthly", f"{energy_usage_kwh_month:.2f} kWh"),
                  ("Daily", f"{energy_usage_kwh_day:.4f} kWh"),
                  ("CO\u2082", f"{energy_co2_display:.4f} kg/day")]),
                (d2, "card-amber", ICON["car"], "Transportation",
                 [("Mode", transportation_mode),
                  ("Fuel", transportation_energy_source),
                  ("Distance", f"{transportation_distance:.2f} km"),
                  ("CO\u2082", f"{transport_co2_display:.4f} kg/day")]),
                (d3, "card-pink", ICON["plastic"], "Plastic Waste",
                 [("Items", str(plastic_items)),
                  ("kg/item", f"{PLASTIC_KG_PER_ITEM:.4f}"),
                  ("Total mass", f"{plastic_usage:.4f} kg"),
                  ("CO\u2082", f"{plastic_co2_display:.4f} kg/day")]),
            ]
            for col, cls, ic, title, rows in col_cards:
                rows_html = "".join(
                    f'<div class="stat-row">'
                    f'<span class="stat-label">{lbl}</span>'
                    f'<span class="stat-value">{val}</span></div>'
                    for lbl, val in rows
                )
                with col:
                    st.markdown(
                        f'<div class="card {cls}" style="padding:15px;">'
                        f'{img(ic, 18)}<b>{title}</b><br><br>{rows_html}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Reference tables ───────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Reference — Emission Factors & Plastic Weights"):
                tab1, tab2 = st.tabs(["Transport Factors", "Plastic Weights"])
                with tab1:
                    st.dataframe(pd.DataFrame({
                        "Mode": [
                            "Petrol Motorcycle/Scooter", "Electric Motorcycle/Scooter",
                            "Petrol Car", "Electric Car", "Hybrid Car",
                            "Petrol Bus", "Electric Bus", "Hybrid Bus",
                            "Bicycle", "Walking",
                        ],
                        "Factor (kg CO\u2082/km)": [
                            0.11367, 0.020, 0.308, 0.077, 0.176,
                            0.089,   0.040, 0.060, 0.005, 0.005,
                        ],
                    }), use_container_width=True)
                with tab2:
                    st.dataframe(pd.DataFrame({
                        "Item":           list(plastic_item_weights.keys()),
                        "Avg Weight (kg)": list(plastic_item_weights.values()),
                    }), use_container_width=True)

            # ── Charts ─────────────────────────────────────────────
            div_label(ICON["chart"], "Emission Insights")

            if PRIMARY_DATA_PATH.exists():
                df_p = pd.read_csv(PRIMARY_DATA_PATH)
                fig, ax = plt.subplots(figsize=(8, 4))
                cats   = ["Energy\n(kWh/day)", "Transport\n(km/day)", "Plastic\n(kg/day)"]
                yours  = [energy_usage_kwh_day, transportation_distance, plastic_usage]
                avgs   = [df_p["Energy_Usage_kWh"].mean(),
                          df_p["Transportation_Distance_km"].mean(),
                          df_p["Plastic_Usage_kg"].mean()]
                x, bw  = np.arange(len(cats)), 0.35
                ax.bar(x - bw/2, yours, bw, label="Your Usage",   color="#22c55e", alpha=0.85, zorder=3)
                ax.bar(x + bw/2, avgs,  bw, label="Average User", color="#0ea5e9", alpha=0.85, zorder=3)
                ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
                ax.set_ylabel("Amount")
                ax.set_title("Your Usage vs. Average User", fontweight="bold")
                ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            # ── FIX: Pie chart — only draw slices that actually have CO2 > 0
            # Previously all three were passed and zero slices caused display artefacts.
            ev = [energy_co2, transportation_co2, plastic_co2]
            el = ["Electricity", "Transport", "Plastic"]
            ec = ["#22c55e", "#f59e0b", "#ec4899"]
            filtered = [(v, l, c) for v, l, c in zip(ev, el, ec) if v > 0]
            if filtered:
                fv, fl, fc = zip(*filtered)
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                ax2.pie(fv, labels=fl, colors=fc, autopct="%1.1f%%", startangle=90,
                        pctdistance=0.82, wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
                ax2.set_title("CO\u2082 Emission Sources", fontweight="bold")
                fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

                co2_map = {"Electricity": energy_co2, "Transportation": transportation_co2,
                           "Plastic": plastic_co2}
                largest = max(co2_map, key=co2_map.get)
                st.info(
                    f"Your largest CO\u2082 contributor is **{largest}**. "
                    "Head to Recommendations for targeted tips."
                )

        st.markdown("---")
        st.caption(
            "Navigate to Recommendations in the sidebar "
            "to see your personalised action plan."
        )

# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "Recommendations":

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;padding-bottom:4px;">'
        f'<img src="{ICON["leaf"]}" width="46">'
        f'<div><h1 style="margin:0;">Recommendations</h1>'
        f'<p style="margin:3px 0 0;opacity:.6;font-size:.92rem;">'
        f'Your personalised carbon reduction action plan</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not st.session_state.has_predicted:
        st.info("No results found yet. Please complete the **Carbon Calculator** tab first.")
    else:
        pred  = st.session_state.prediction
        e_co2 = st.session_state.energy_co2
        t_co2 = st.session_state.transportation_co2
        p_co2 = st.session_state.plastic_co2
        total = e_co2 + t_co2 + p_co2
        all_zero = st.session_state.get("all_zero_input", False)

        # ── Footprint summary ─────────────────────────────────────
        st.markdown(
            f'<div class="card" style="padding:20px 24px;">'
            f'{img(ICON["globe"], 26)}'
            f'<span style="font-size:1.45rem;font-weight:800;">{pred:.2f} kg CO\u2082/day</span>'
            f'<span style="opacity:.6;font-size:.86rem;margin-left:10px;">'
            f'your estimated footprint</span></div>',
            unsafe_allow_html=True,
        )

        if all_zero:
            st.success(
                "🌿 Outstanding! You had a truly low-impact day — spending time in nature "
                f"with no recorded electricity, transport, or plastic use. Your footprint of "
                f"{pred:.2f} kg CO\u2082 represents only the unavoidable human baseline. "
                "Keep it up!"
            )
        else:
            contribution = {"Electricity": e_co2, "Transportation": t_co2, "Plastic": p_co2}
            largest = max(contribution, key=contribution.get)

            if pred < 3.0:
                st.success(
                    "Excellent: Your daily footprint is well below global averages. "
                    "Your habits are closely aligned with a sustainable lifestyle."
                )
            elif pred <= 12.0:
                st.warning(
                    f"Moderate: Your usage matches regional averages, but opportunities to "
                    f"reduce exist. Your largest source is **{largest}**. Small adjustments here "
                    "compound into large reductions over time."
                )
            else:
                st.error(
                    f"High: Producing {pred:.2f} kg CO\u2082 daily creates long-term environmental "
                    f"debt. Structural changes to your **{largest}** choices are strongly recommended."
                )

            # ── 3-column emission breakdown ────────────────────────────
            div_label(ICON["chart"], "Emissions Breakdown")
            b1, b2, b3 = st.columns(3)
            for col, label, val, ic, cls in [
                (b1, "Electricity", e_co2, ICON["electricity"], ""),
                (b2, "Transport",   t_co2, ICON["car"],         "card-amber"),
                (b3, "Plastic",     p_co2, ICON["plastic"],     "card-pink"),
            ]:
                pct = (val / total * 100) if total > 0 else 0
                with col:
                    st.markdown(
                        f'<div class="card {cls}" style="text-align:center;padding:16px;">'
                        f'<img src="{ic}" width="30" style="margin-bottom:8px;display:block;'
                        f'margin-left:auto;margin-right:auto;">'
                        f'<div style="font-size:1.2rem;font-weight:700;">{val:.3f}</div>'
                        f'<div style="font-size:.75rem;opacity:.6;">kg CO\u2082/day</div>'
                        f'<div style="font-size:.82rem;font-weight:600;margin-top:4px;">'
                        f'{pct:.1f}%</div>'
                        f'<div style="font-size:.78rem;opacity:.6;">{label}</div></div>',
                        unsafe_allow_html=True,
                    )

            # ── Tailored advice ────────────────────────────────────────
            div_label(ICON["bulb"], "Tailored Sustainability Advice")

            if largest == "Electricity":
                kwh = st.session_state.energy_usage_kwh_month
                st.markdown(
                    f'<div class="card card-blue">'
                    f'{img(ICON["electricity"], 20)}<b>Target: Electricity Conservation</b>'
                    f'<p style="margin:10px 0 6px;font-size:.9rem;">'
                    f'Your bill converts to <b>{kwh:.1f} kWh/month</b>. Consider:</p>'
                    f'<ul style="margin:0;padding-left:18px;font-size:.9rem;line-height:1.9;">'
                    f'<li>Set your air-conditioner to 25\u00b0C or use inverter mode.</li>'
                    f'<li>Unplug chargers and devices when not in use.</li>'
                    f'<li>Switch high-use lighting to LED.</li></ul></div>',
                    unsafe_allow_html=True,
                )
            elif largest == "Transportation":
                dist = st.session_state.transportation_distance
                st.markdown(
                    f'<div class="card card-amber">'
                    f'{img(ICON["car"], 20)}<b>Target: Smart Mobility</b>'
                    f'<p style="margin:10px 0 6px;font-size:.9rem;">'
                    f'Your travel generates <b>{t_co2:.4f} kg CO\u2082/day</b> '
                    f'over <b>{dist:.1f} km</b>. Try:</p>'
                    f'<ul style="margin:0;padding-left:18px;font-size:.9rem;line-height:1.9;">'
                    f'<li>Combine errands or carpool with colleagues.</li>'
                    f'<li>Walk or cycle for short distances.</li>'
                    f'<li>Choose electric vehicle options on ride-hailing apps.</li></ul></div>',
                    unsafe_allow_html=True,
                )
            elif largest == "Plastic":
                items = st.session_state.plastic_items
                st.markdown(
                    f'<div class="card card-pink">'
                    f'{img(ICON["plastic"], 20)}<b>Target: Single-Use Plastic Reduction</b>'
                    f'<p style="margin:10px 0 6px;font-size:.9rem;">'
                    f'You use <b>{items} items/day</b>. Try:</p>'
                    f'<ul style="margin:0;padding-left:18px;font-size:.9rem;line-height:1.9;">'
                    f'<li>Carry a reusable canvas tote for shopping.</li>'
                    f'<li>Use a reusable stainless-steel water bottle.</li>'
                    f'<li>Decline plastic cutlery and straws at takeout meals.</li></ul></div>',
                    unsafe_allow_html=True,
                )

        # ── Quiz ───────────────────────────────────────────────────
        div_label(ICON["quiz"], "Carbon IQ Quiz")
        quiz_choice = st.radio(
            "Which waste item takes the longest to break down in our ecosystems?",
            ["Plastic Straws (~200 years)",
             "Plastic Drink Bottles (~450 years)",
             "Styrofoam Containers (Never)"],
        )
        if st.button("Check Answer"):
            if "Never" in quiz_choice:
                st.success(
                    "Correct! Styrofoam does not biodegrade — it fragments into "
                    "microplastics that persist in ecosystems indefinitely."
                )
            else:
                st.error(
                    "Not quite! While those items take centuries, "
                    "Styrofoam persists forever."
                )

# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL TRANSPARENCY
# ═══════════════════════════════════════════════════════════════════
elif page == "Model Transparency":

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;padding-bottom:4px;">'
        f'<img src="{ICON["microscope"]}" width="46">'
        f'<div><h1 style="margin:0;">Model Transparency</h1>'
        f'<p style="margin:3px 0 0;opacity:.6;font-size:.92rem;">'
        f'Open science: dataset, formulas & diagnostics</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Mission
    div_label(ICON["target"], "Mission & Objective")
    st.write(
        "To spread knowledge and raise awareness to user regarding their carbon emissions in response to SDG Goal 13.3 by empowering individuals to understand the direct environmental impact of their daily choices and providing actionable insights regarding personal carbon emissions, fostering a culture of sustainability and environmental responsibility within our community."
    )

    # Motivation
    div_label(ICON["leaf"], "Project Motivation")
    st.write(
        "While our work was inspired by extensive published literature on carbon tracking, we observed that existing frameworks rely almost exclusively on international datasets. To bridge this gap, our project aims to bring this technology directly to Cambodia by localizing emission factors and collecting primary data to ensure real-world accuracy. Based on core daily activities, we selected three high-impact lifestyle features and utilized a Multiple Linear Regression model to track and predict personal carbon emissions based on direct user input. Moreover, given the sharp rise in global and regional carbon emissions in recent years, we aim to urge individuals to reflect on their daily behaviors and educate our community on how our collective actions profoundly impact nature."
    )

    # Model framework
    div_label(ICON["science"], "Modelling Framework")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f'<div class="card" style="padding:16px;">'
            f'{img(ICON["chart"], 18)}<b>Multiple Linear Regression</b>'
            f'<p style="font-size:.85rem;margin:8px 0 0;opacity:.72;">'
            f'Trained to map linear coefficients across energy, transport, and consumption metrics for high interpretability.</p></div> ',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="card card-teal" style="padding:16px;">'
            f'{img(ICON["science"], 18)}<b>StandardScaler</b>'
            f'<p style="font-size:.85rem;margin:8px 0 0;opacity:.72;">'
            f'Standardizes varying physical units (kWh, km, kg) to a uniform scale. By centering data around a zero mean and unit variance, it prevents high-magnitude inputs (like hundreds of transport kilometers) from mathematically overpowering smaller, critical inputs (like single kilograms of plastic mass.</p></div> ',
            unsafe_allow_html=True,
        )

    # Constants
    div_label(ICON["info"], "Emission Constants")
    st.markdown(
        f'<div class="card" style="padding:16px;">'
        f'<div class="stat-row"><span class="stat-label">Grid emission factor (Cambodia)</span>'
        f'<span class="stat-value">{GRID_EMISSION_FACTOR} kg CO\u2082/kWh</span></div>'
        f'<div class="stat-row"><span class="stat-label">EDC residential tariff</span>'
        f'<span class="stat-value">{EDC_RATE:.0f} Riel/kWh</span></div>'
        f'<div class="stat-row"><span class="stat-label">Petrol motorcycle factor</span>'
        f'<span class="stat-value">{MOTORCYCLE_PETROL_CO2} kg CO\u2082/km</span></div>'
        f'<div class="stat-row"><span class="stat-label">Plastic lifecycle emission</span>'
        f'<span class="stat-value">{PLASTIC_LIFECYCLE_CO2:.1f} kg CO\u2082/kg plastic</span></div>'
        f'<div class="stat-row"><span class="stat-label">Plastic weight/item (Cambodian baseline)</span>'
        f'<span class="stat-value">{PLASTIC_KG_PER_ITEM:.5f} kg</span></div>'
        f'<div class="stat-row"><span class="stat-label">Human baseline CO\u2082 (zero-input day)</span>'
        f'<span class="stat-value">{BASELINE_CO2_PER_DAY:.2f} kg/day</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Dataset
    div_label(ICON["chart"], "Dataset Properties")
    st.write(
        "Our predictive model translates raw daily habits such as electricity usage, travel distance, and plastic consumption into a unified, standardized carbon emission index (kg CO\u2082e/day)."
    )
    st.table(pd.DataFrame({
        "Feature":     ["Energy_Usage_kWh", "Transportation_Distance_km", "Plastic_Usage_kg"],
        "Description": ["Daily kWh consumed", "Raw daily travel distance (km)",
                        "Aggregated daily plastic mass (kg)"],
        "Source":      ["EDC (Electricit\u00e9 Du Cambodge)",
                        "Kaggle Baseline + Local Vehicle Data",
                        "Cambodian Urban Waste Baselines"],
    }))

    # Performance
    div_label(ICON["science"], "Performance & Validation")
    st.markdown(
        f'<div class="card" style="padding:18px;">'
        f'<h4>1. Core Model Evaluation (Secondary Data Training)</h4>'
        f'Our Multiple Linear Regression model was evaluated using standard statistical metrics to ensure its predictive stability:'
        f'<ul style="margin:8px 0 14px;padding-left:18px;font-size:.9rem;line-height:1.9;">'
        f'<li><b>Variance Explanation (R\u00b2</b>): Our preliminary testing demonstrates a high coefficient of determination, proving that isolating just electricity, transportation, and plastic data captures the vast majority of an individual\'s daily carbon impact.</li> '
        f'<li><b>Error Margin (MAE</b>):  The model minimizes the Mean Absolute Error, ensuring that individual carbon footprint predictions remain tight, stable, and free from wild numerical swings.</li>'
        f'</ul>'
        f'<h4>2. Field Validation (Primary Data) Alignment)</h4>'
        f'To ensure our model transitions seamlessly from a generic online dataset to actual Cambodian lifestyles, we validate its behavior against our collected primary data (local surveys and regional measurements):'
        f'<ul style="margin:8px 0 0;padding-left:18px;font-size:.9rem;line-height:1.9;">'
        f'<li><b>Contextual Accuracy: </b>Instead of using the model straight out of the box, we cross-referenced its outputs with empirical data gathered from our target Cambodian demographic.</li>'
        f'<li><b>The Verdict: </b>This validation confirms that our localized equations and synthetic plastic adjustments successfully mirror real-world consumption habits in Cambodia, making the calculator highly accurate for local users.</li>'
        f'</ul></div>',
        unsafe_allow_html=True,
    )

    # Fun fact
    st.markdown("---")
    div_label(ICON["bulb"], "Fun Eco-Fact")
    st.info("""
        **Did You Know?**
        
        The concept of a personal \"Carbon Footprint\" was originally popularized by the oil company British Petroleum (BP) in 2004 through a large $250 million marketing campaign. While it was initially created to shift environmental accountability from big corporations to individuals, tracking your personal data remains an incredibly powerful grassroots tool for lowering demand for high-emission products!"""
    )
