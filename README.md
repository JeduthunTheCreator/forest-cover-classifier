# Forest Cover Type Classifier
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Tensorflow-2.16+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.14.0-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0.1-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.6-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.0-11557c?style=flat&logoColor=white)](https://matplotlib.org/)

A deep learning classification model built with TensorFlow/Keras that identifies the dominant forest cover type across a 30x30 meter cell of land in the Roosevelt National Forest, Colorado, based purely on cartographic variables.

## Project Overview
This project builds and optimizes a series of neural network classifiers to predict 
forest cover type across 30x30 meter cells of land in the Roosevelt National Forest, 
Colorado. Three models are developed and compared — a baseline classifier, an improved 
architecture with dropout regularization and adaptive learning, and an ensemble model 
combining both — trained on the UCI Forest Cover dataset sourced from Kaggle. The 
project explores key deep learning concepts including class imbalance handling via 
computed class weights, hyperparameter tuning across architecture depth and learning 
rates, and ensemble prediction strategies. Model predictions are interpreted using 
SHAP (SHapley Additive exPlanations), providing global feature importance rankings 
and individual prediction explanations to understand what cartographic variables drive 
each cover type classification. All training diagnostics, confusion matrices and model 
comparisons are visualised using Matplotlib and Seaborn, with findings and conclusions 
documented across dedicated analysis and interpretability notebooks.

---

## Notebooks
| Notebook | Description |
|----------|-------------|
| `forest_cover_eda.ipynb` | Exploratory data analysis — distributions, correlations, class imbalance |
| `forest_cover_analysis.ipynb` | Full pipeline — preprocessing, training, evaluation, ensemble |
| `shap_analysis.ipynb` | Model interpretability — global and per-class feature importance via SHAP |

---

## Live Demo

<!--[Link to deployed Streamlit app]  ← if you deploy it on Streamlit Cloud -->


### Streamlit App
streamlit run app.py

---

## Dataset
**Source:** [UCI Machine Learning Repository — Forest Cover Type Dataset](https://archive.ics.uci.edu/dataset/31/covertype)

### Handling Class Imbalance
The dataset is heavily imbalanced — Lodgepole Pine and Spruce/Fir together 
account for over 85% of samples. Balanced class weights are computed and 
applied during training, penalising misclassification of rare species 
proportionally to their scarcity.

### Dataset Features

| Feature | Type | Description |
|---|---|---|
| Elevation | Continuous | Elevation in meters |
| Aspect | Continuous | Aspect in degrees azimuth |
| Slope | Continuous | Slope in degrees |
| Horizontal_Distance_To_Hydrology | Continuous | Horizontal distance to nearest water feature (meters) |
| Vertical_Distance_To_Hydrology | Continuous | Vertical distance to nearest water feature (meters) |
| Horizontal_Distance_To_Roadways | Continuous | Horizontal distance to nearest roadway (meters) |
| Hillshade_9am | Continuous | Hillshade index at 9am summer solstice (0–255) |
| Hillshade_Noon | Continuous | Hillshade index at noon summer solstice (0–255) |
| ~~Hillshade_3pm~~ | ~~Continuous~~ | ~~Hillshade index at 3pm summer solstice (0–255) — dropped due to high correlation with Hillshade_9am~~ |
| Horizontal_Distance_To_Fire_Points | Continuous | Horizontal distance to nearest wildfire ignition point (meters) |
| Wilderness_Area (4 columns) | Binary | Wilderness area designation (0 = absent, 1 = present) |
| Soil_Type (40 columns) | Binary | Soil type designation (0 = absent, 1 = present) |
| **Cover_Type** | **Target** | **Forest cover type — 7 classes** |

### Target Classes

| Class | Cover Type |
|---|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

---

## Model Architecture
### Baseline model

```
Input Layer  →  53 features
Dense(64)    →  ReLU activation
Dense(32)    →  ReLU activation
Output(7)     →  Softmax activation
```

**Training configuration:**
- Optimiser: Adam(default learning rate=0.001)
- Batch size: 32
- Max epochs: 150
- Early stopping: patience of 10 on validation loss, restoring best weights
- Validation: explicit stratified split(20% of training data)

### Training Diagonistics
The classification report reveals a consistent pattern across minority classes: **high recall, low precision**. The model successfully finds most instances of rare classes but casts too wide a net, pulling in many misclassified samples alongside the correct ones. This is the expected consequence of aggressive class weighting.

<!--**insert plots here**-->


### Improved model
```
Input Layer  →  53 features
Dense(256)    →  ReLU activation + Dropout (0.2)
Dense(128)    →  ReLU activation + Dropout (0.2)
Dense(64)     →  ReLU activation + Dropout (0.1)
Output(7)     →  Softmax activation
```

**Training configuration:**
- Optimiser: Adam(default learning rate=0.0005)
- Batch size: 32
- Max epochs: 150
- Early stopping: patience of 10 on validation loss, restoring best weights
- ReduceLROnPlateu: halves learning rate after 4 epochs without improvement, minimum learning rate 1e-6
- Validation: explicit stratified split(20% of training data)

### Training Diagnostics

<!--**insert write up and plots** -->

---

## Results Summary
| Model       | Accuracy | Macro F1 |
|-------------|----------|----------|
| Baseline    | 0.81     | 0.76     |
| Improved    | 0.85     | 0.80     |
| Ensemble    | 0.84     | 0.79     |

The improved model outperforms the baseline across 6 of 7 classes. 
The ensemble did not improve on the improved model alone — consistent 
with theory when one model is significantly stronger than the other.

---

## Installation
### Prerequisites
```bash
pip install -r requirements.txt
```

1. Clone the repository
```bash
git clone <your-repo-url>
cd forest-cover-classifier
```

2. Download the dataset and place the cover_data.csv in the project root directory

3. Train the model 
```bash
python src/baseline_script.py --filepath cover_data.csv
```

---

## Sample command to run Python Script in CLI
### Baseline:
```bash
python baseline_script.py --filepath ../cover_data.csv --epochs 150 --batch-size 32 --test-size 0.2 --random-state 42
```
### Improved:
```bash
python improved_script.py --filepath ../cover_data.csv --epochs 150 --batch-size 32 --test-size 0.2 --random-state 42
```
### Ensemble:
```bash
python ensemble_script.py --filepath ../cover_data.csv --baseline-model ../output/forestcover_baseline_acc0.850_ts0.2_rs42_20260520_120000.keras --improved-model ../output/forestcover_improved_acc0.900_ts0.2_rs42_20260520_130000.keras --test-size 0.2 --random-state 42
```

---

## Project Structure
```bash
forest-cover-classifier/
├── notebooks/
│   ├── forest_cover_analysis.ipynb
│   ├── forest_cover_eda.ipynb
│   ├── shap_analysis.ipynb
│
├── output/               ← saved models and plots
├── app.py                ← Streamlit app
│
├── src/
│   ├── baseline_script.py
│   ├── improved_script.py
│   ├── ensemble_script.py
│   └── utils.py
│ 
└── requirements.txt
```

---

## Key Findings
- The improved model (deeper architecture with dropout) outperforms the baseline 
  across 6 of 7 classes, achieving 85% accuracy and a macro F1 of 0.80 vs 0.76
- Class weighting consistently produces high recall but low precision for minority 
  classes — Aspen precision peaked at 0.41 across all models, a structural 
  consequence of its class weight of 8.74 rather than an architectural limitation
- A fair epoch budget with early stopping as the convergence criterion is essential 
  for meaningful model comparison — at 50 epochs the improved model appeared worse 
  than the baseline simply because it had not yet converged at the lower learning rate
- The ensemble did not outperform the improved model alone — consistent with theory 
  when one model is significantly stronger than the other, averaging in the weaker 
  model's predictions dilutes rather than corrects the stronger model's decisions
- Elevation is the most influential feature across all seven cover types according 
  to SHAP, reflecting the distinct elevation bands each tree species occupies in 
  the Roosevelt National Forest
- The dominant class confusion between Spruce/Fir and Lodgepole Pine is the single 
  largest source of error across all models, driven by their overlapping cartographic 
  profiles and combined 85% share of the dataset

---

## Concepts Demonstrated
- Multi-class neural network classification with the Keras Sequential API
- Class imbalance handling via balanced class weight computation
- Data leakage prevention — scaler fitted on training data only, explicit stratified 
  validation splits replacing Keras's uncontrolled validation_split parameter
- Regularisation via Dropout across multiple layers
- Early stopping with best weight restoration and ReduceLROnPlateau scheduling
- Ensemble methods — simple averaging and validation-optimised weighted averaging 
  using scipy's L-BFGS-B minimiser
- Model evaluation with classification report, confusion matrix and macro F1 score
- Mixed feature preprocessing with ColumnTransformer — StandardScaler applied to 
  continuous features only, binary features passed through unscaled
- Model interpretability with SHAP — global feature importance, per-class analysis, 
  individual prediction explanations and feature dependence plots
- Training curve visualisation and multi-model comparison with Matplotlib and Seaborn

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
