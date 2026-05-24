import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

from tensorflow.keras.models import load_model
from utils import CLASS_LABELS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Forest Cover Classifier',
    page_icon='🌲',
    layout='wide'
)

# ── Load model and scaler once ────────────────────────────────────────────────
# st.cache_resource ensures the model is loaded only once, not on every user interaction

# Build absolute paths relative to app.py's location
APP_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../src/
ROOT_DIR  = os.path.dirname(APP_DIR)                     # .../forest-cover-classifier/

MODEL_PATH  = os.path.join(ROOT_DIR, 'output', 'forestcover_model_acc0.851_ts0.2_rs42_20260521_204708.keras')
SCALER_PATH = os.path.join(ROOT_DIR, 'output', 'scaler.joblib')

BASELINE_CM     = os.path.join(ROOT_DIR, 'output', 'baseline_model_confusion_matrix.png')
IMPROVED_CM     = os.path.join(ROOT_DIR, 'output', 'improved_model_confusion_matrix.png')
BASELINE_CURVES = os.path.join(ROOT_DIR, 'output', 'baseline_model_training_curves.png')
IMPROVED_CURVES = os.path.join(ROOT_DIR, 'output', 'improved_model_training_curves.png')
COMPARISON_PLOT = os.path.join(ROOT_DIR, 'output', 'ensemble_model_comparison.png')


@st.cache_resource
def load_artifacts():
    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_artifacts()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['🌲 Predict', '📊 Model Performance', 'ℹ️ About'])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title('Forest Cover Type Predictor')
    st.markdown('Adjust the terrain features below to predict the dominant '
                'tree species for a 30×30m area of the Roosevelt National Forest.')

    st.subheader('Terrain Features')
    col1, col2, col3 = st.columns(3)

    with col1:
        elevation = st.slider('Elevation (m)',          1800, 3900, 2500)
        aspect    = st.slider('Aspect (degrees)',          0,  360,  180)
        slope     = st.slider('Slope (degrees)',            0,   66,   15)

    with col2:
        h_hydro = st.slider('Horizontal Distance to Hydrology (m)',  0, 1400, 200)
        v_hydro = st.slider('Vertical Distance to Hydrology (m)',  -170,  600,  30)
        h_road  = st.slider('Horizontal Distance to Roadways (m)',    0, 7000, 1500)

    with col3:
        hillshade_9am  = st.slider('Hillshade 9am',    0, 255, 200)
        hillshade_noon = st.slider('Hillshade Noon',   0, 255, 220)
        h_fire         = st.slider('Horizontal Distance to Fire Points (m)', 0, 7000, 1500)

    st.subheader('Wilderness Area & Soil Type')
    col4, col5 = st.columns(2)

    with col4:
        wilderness = st.selectbox('Wilderness Area', [
            'Rawah', 'Neota', 'Comanche Peak', 'Cache la Poudre'
        ])

    with col5:
        soil_type = st.selectbox('Soil Type', [f'Soil Type {i+1}' for i in range(40)])

    # ── Build feature vector ──────────────────────────────────────────────────
    if st.button('Predict Cover Type', type='primary', use_container_width=True):

        # Continuous features
        continuous = [elevation, aspect, slope, h_hydro, v_hydro,
                      h_road, hillshade_9am, hillshade_noon, h_fire]

        # Binary wilderness area (4 features)
        wilderness_options = ['Rawah', 'Neota', 'Comanche Peak', 'Cache la Poudre']
        wilderness_binary  = [1 if wilderness == w else 0 for w in wilderness_options]

        # Binary soil type (40 features)
        soil_idx    = int(soil_type.split()[-1]) - 1
        soil_binary = [1 if i == soil_idx else 0 for i in range(40)]

        # Combine into DataFrame with correct column names for the scaler
        continuous_features = [
            'Elevation', 'Aspect', 'Slope',
            'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon',
            'Horizontal_Distance_To_Fire_Points'
        ]
        wilderness_cols = ['Wilderness_Area1','Wilderness_Area2',
                           'Wilderness_Area3','Wilderness_Area4']
        soil_cols       = [f'Soil_Type{i+1}' for i in range(40)]

        input_df = pd.DataFrame(
            [continuous + wilderness_binary + soil_binary],
            columns=continuous_features + wilderness_cols + soil_cols
        )

        # Scale and predict
        input_scaled = np.asarray(scaler.transform(input_df), dtype=np.float32)
        probabilities = model.predict(input_scaled, verbose=0)[0]
        predicted_idx = np.argmax(probabilities)

        # ── Display results ───────────────────────────────────────────────────
        st.divider()
        col6, col7 = st.columns([1, 2])

        with col6:
            st.metric('Predicted Cover Type', CLASS_LABELS[predicted_idx])
            st.metric('Confidence', f'{probabilities[predicted_idx]*100:.1f}%')

        with col7:
            st.subheader('Probability Distribution')
            prob_df = pd.DataFrame({
                'Cover Type': CLASS_LABELS,
                'Probability': probabilities
            }).sort_values('Probability', ascending=True)

            st.bar_chart(prob_df.set_index('Cover Type'))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title('Model Performance')

    st.subheader('Results Summary')
    results_df = pd.DataFrame({
        'Model':       ['Baseline', 'Improved', 'Ensemble'],
        'Accuracy':    [0.81, 0.85, 0.84],
        'Macro F1':    [0.76, 0.80, 0.79],
        'Loss':        [0.4632, 0.3651, '—']
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.divider()

    col8, col9 = st.columns(2)
    with col8:
        st.subheader('Baseline — Confusion Matrix')
        st.image(BASELINE_CM)

    with col9:
        st.subheader('Improved — Confusion Matrix')
        st.image(IMPROVED_CM)

    st.subheader('Training Curves Comparison')
    col10, col11 = st.columns(2)
    with col10:
        st.image(BASELINE_CM)
    with col11:
        st.image(IMPROVED_CM)

    st.subheader('Model Comparison')
    st.image(COMPARISON_PLOT, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title('About This Project')
    st.markdown("""
    This application is the deployment component of a deep learning project 
    that classifies forest cover types across 30×30 metre cells in the 
    Roosevelt National Forest, Colorado, using cartographic variables only.

    **Three models were developed and compared:**
    - **Baseline** — shallow 2-layer network (64 → 32 → 7)
    - **Improved** — deeper regularised network with Dropout and adaptive LR
    - **Ensemble** — weighted combination of both models

    The best performing model (Improved, 85.1% accuracy) is used for predictions 
    in this app.

    **Dataset:** [UCI Covertype Dataset](https://archive.ics.uci.edu/dataset/31/covertype) 
    — 581,012 samples, 7 classes, 54 original features

    **Tech stack:** TensorFlow · Keras · Pandas · NumPy · Matplotlib · Streamlit 

    📓 [View full analysis notebook](https://github.com/JeduthunTheCreator/forest-cover-classifier/blob/main/notebooks/forest_cover_analysis.ipynb) | 💻 [GitHub Repository](https://github.com/JeduthunTheCreator/forest-cover-classifier)
    """)
