# 🚚 Driver Churn Prediction API

A production-ready Machine Learning project that predicts whether a driver is likely to churn using a **Random Forest Classifier**. The project demonstrates the complete ML lifecycle—from data preprocessing and model training to hyperparameter tuning, model persistence, and serving predictions through a FastAPI REST API.

---

# 🚀 Features

- Train a Random Forest Classifier
- Evaluate model performance
- Perform 5-Fold Cross Validation
- Tune hyperparameters using GridSearchCV
- Save trained model using Joblib
- Load trained model without retraining
- REST API built with FastAPI
- Interactive API documentation (Swagger UI)
- Predict churn for new drivers

---

# 📚 Machine Learning Concepts Covered

- Data Loading
- Feature Selection
- Train/Test Split
- Random Forest Classification
- Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report
- Cross Validation
- Hyperparameter Tuning
- Feature Importance
- Model Serialization
- Model Deployment

---

# 🛠 Tech Stack

## Machine Learning

- Python
- Pandas
- Scikit-Learn
- Joblib

## Backend

- FastAPI
- Uvicorn
- Pydantic

---

# 📂 Project Structure

```text
driver-churn-api/
│
├── data/
│   └── driver_churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── train.py          # Train and save model
├── app.py            # FastAPI server
├── requirements.txt
├── README.md
└── venv/
```

---

# 📊 Dataset

Features:

| Column | Description |
|----------|-------------|
| loads_completed | Number of completed loads |
| last_login_days | Days since last login |
| rating | Driver rating |

Target:

| Column | Description |
|----------|-------------|
| churn | 0 = Driver Stays |
| churn | 1 = Driver Leaves |

---

# 🌳 Model Training

The project uses a Random Forest Classifier.

```python
model = RandomForestClassifier(
    random_state=42
)
```

The model is trained using historical driver data and then saved for future predictions.

---

# 🔄 Cross Validation

Model performance is evaluated using **5-Fold Cross Validation**.

Benefits:

- More reliable evaluation
- Reduces dependence on a single train/test split
- Better estimate of real-world performance

Example:

```
Fold 1 → Train | Test
Fold 2 → Train | Test
Fold 3 → Train | Test
Fold 4 → Train | Test
Fold 5 → Train | Test
```

---

# ⚙ Hyperparameter Tuning

The project uses **GridSearchCV** to automatically search for the best Random Forest configuration.

Example search space:

```python
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [2, 4, None],
    "min_samples_split": [2, 4]
}
```

The best-performing model is selected based on cross-validation accuracy.

---

# 📈 Model Evaluation

Evaluation metrics include:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Cross Validation Accuracy

These metrics help measure how well the model predicts driver churn.

---

# 💾 Model Persistence

Once training is complete, the model is saved using Joblib.

```python
joblib.dump(model, "models/churn_model.pkl")
```

Later, the API loads the trained model without retraining.

```python
model = joblib.load("models/churn_model.pkl")
```

This significantly reduces startup time and enables efficient inference.

---

# 🌐 FastAPI Deployment

The project exposes the trained model through a REST API.

Start the server:

```bash
uvicorn app:app --reload
```

Server:

```
http://127.0.0.1:8000
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

# 📡 API Endpoint

## POST /predict

Request:

```json
{
  "loads_completed": 6,
  "last_login_days": 25,
  "rating": 3.6
}
```

Example Response:

```json
{
  "churn": 1
}
```

---

# 🔍 Prediction Flow

```text
Driver Data
      │
      ▼
FastAPI Endpoint
      │
      ▼
Pydantic Validation
      │
      ▼
Load Trained Model
      │
      ▼
Random Forest Prediction
      │
      ▼
Return JSON Response
```

---

# 📚 Skills Demonstrated

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Hyperparameter tuning
- Cross validation
- Model persistence
- REST API development
- Pydantic request validation
- Production-ready inference pipeline

---

# 🚀 Future Improvements

- Return prediction probability
- Feature importance endpoint
- Batch prediction endpoint
- Docker support
- CI/CD pipeline
- Unit and integration tests
- Model versioning
- Authentication and rate limiting
- Deploy to a cloud platform (AWS, Azure, or Google Cloud)

---

# 🎯 Learning Outcome

This project demonstrates the complete workflow of a production-ready machine learning application:

1. Prepare data
2. Train a model
3. Evaluate performance
4. Tune hyperparameters
5. Save the trained model
6. Build a REST API
7. Serve real-time predictions

It serves as a foundation for deploying machine learning models into real-world applications.
