# Carbon-Footprint-Emission-Analytics

## Localized Carbon Footprint Predictive Dashboard
An interactive, data-driven web application designed to estimate personal lifestyle carbon footprints within the specific context of the population in Cambodia.

This project bridges the regional data gap by training an optimized predictive engine on global secondary lifestyle data and adapting it seamlessly to local realities using robust standardization pipelines and primary data collected from Cambodian citizens.

## Core Features

* **Predictive Calculator:** Translates accessible daily metrics (e.g., monthly utility billing tiers, single-use plastic item counts, and commuting distances) into localized daily greenhouse gas estimates (kgCO2).
* **Model Transparency Tab:** Opens the machine learning black box by visualizing real-time beta ($\beta$) coefficients and the mathematical hyperplane equation directly for the user.
* **Interactive Optimization Sliders:** A green action recommendation module that simulates how altering specific habits dynamically offsets individual carbon output.

## Technical Architecture & Metrics

* **Core Architecture:** Multivariate Linear Regression (selected over Random Forest for optimal explainability, low computational latency, and user-facing transparency).
* **Baseline Performance (Global Training):** $R^2 = 0.8537$ | $\text{MAE} = 2.9561\text{ kg}$
* **Local Validation Performance (Cambodian Students):** Exceptional validation $R^2 = 0.9368$ | $\text{MAE} = 0.4300\text{ kg}$
* **Data Engineering:** Features an active preprocessing layer utilizing a pre-trained `StandardScaler` to map regional input variations into unitless Z-scores, preserving predictive integrity across socioeconomic boundaries.
* **Primary Driver Identification:** Quantified single-use plastic waste generation as the dominant structural driver of emission variance within the model space ($\beta_{\text{Plastic}} = 3.5413$).

## Tech Stack

* **Language:** Python
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Deployment:** Streamlit (Web Dashboard Interface)

---

*Developed in alignment with **UN Sustainable Development Goal 13 (Climate Action)** to improve environmental literacy and foster sustainable behavioral modifications among Cambodian youth.*
