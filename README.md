# Swiggy Delivery Time Prediction System

A complete end-to-end **Machine Learning + MLOps pipeline** designed to predict food delivery time using real-world features such as distance, traffic, weather, and order details.

This project focuses not only on model accuracy but also on building a **production-ready, reproducible, and trackable ML system**.

---

## Project Overview

This system predicts the **time required for food delivery** using structured data.
It integrates multiple stages of the ML lifecycle including:

* Data ingestion and preprocessing
* Feature engineering
* Model training and hyperparameter tuning
* Model evaluation
* Experiment tracking with MLflow
* Data versioning with DVC
* Model registry and staging

---

## Tech Stack

**Programming & Libraries**

* Python
* Pandas, NumPy
* Scikit-learn
* LightGBM, XGBoost

**MLOps Tools**

* MLflow (experiment tracking + model registry)
* DVC (data versioning)
* AWS S3 (remote storage)
* DagsHub (MLflow tracking server)

---

## Project Structure

```bash
swiggy-delivery-time-prediction/
│
├── data/
├── models/
├── src/
├── params.yaml
├── dvc.yaml
├── run_information.json
└── README.md
```

---

## Data Pipeline

### Data Cleaning

* Removed invalid entries and inconsistencies
* Handled missing values
* Extracted date and time features
* Standardized categorical values

---

### Data Preparation

* Train-test split using configurable parameters
* Saved intermediate datasets

---

### Feature Engineering

* MinMaxScaler for numerical features
* OneHotEncoder for categorical features
* OrdinalEncoder for ordered features
* Distance calculation using Haversine formula

---

## Model Development

### Model Exploration

Tested multiple models:

* Random Forest
* LightGBM
* XGBoost
* Gradient Boosting
* KNN

---

### Hyperparameter Tuning

* Used **Optuna** for efficient optimization
* Tuned:

  * Random Forest
  * LightGBM

---

### Final Model Architecture

* **Base Models**

  * Random Forest
  * LightGBM

* **Meta Model**

  * Linear Regression

Combined using a **Stacking Regressor** with target transformation using PowerTransformer.

---

## Evaluation Results

The final model achieved the following performance:

* **Train MAE**: 3.04

* **Test MAE**: 3.19

* **Mean Cross-Validation MAE**: 3.18

* **Train R² Score**: 0.835

* **Test R² Score**: 0.819

### Interpretation

* The model generalizes well (train and test scores are close)
* Low MAE indicates strong prediction accuracy
* Stable cross-validation score confirms robustness

---

## Experiment Tracking

Used **MLflow** for:

* Logging parameters and metrics
* Tracking experiments
* Storing models and artifacts
* Comparing different runs

---

## Model Registry

* Registered model using MLflow
* Promoted best model to **Staging**
* Stored run metadata for reproducibility

---

## Data Versioning

* Used **DVC** for version control
* Connected to AWS S3 for storage
* Ensured reproducibility

---

## Key Highlights

* Built a complete **end-to-end ML pipeline**
* Applied **Optuna-based hyperparameter tuning**
* Designed an **ensemble stacking model**
* Integrated **MLflow for tracking and registry**
* Implemented **DVC for data versioning**
* Structured project for production readiness

---

## How to Run

```bash
git clone https://github.com/rahulpatel0628/swiggy-delivery-time-prediction
cd swiggy-delivery-time-prediction

python -m venv myenv
myenv\Scripts\activate

pip install -r requirements.txt

python src/data_cleaning.py
python src/data_preparation.py
python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py
```

---

## Future Improvements

* Add FastAPI for deployment
* Containerize using Docker
* Implement CI/CD pipeline
* Build UI for real-time predictions

---

## Conclusion

This project demonstrates how to build a **complete ML system** that is scalable, reproducible, and production-ready.

---

## Author

Rahul Patel
B.Tech Student | Aspiring Data Scientist & MLOps Engineer
