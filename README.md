# BMI Prediction 🤖

Predicting Body Mass Index (BMI) using Neural Network and Linear Regression models.

## 📌 About the Project
This project compares two machine learning models — Neural Network and Linear Regression — to predict BMI from a person's weight and height.

## 📊 Dataset
- **Source:** AI Generated
- **Total Samples:** 200
- **Features:**
  - Weight (kg) — range: 40–90
  - Height (m) — range: 1.4–1.9
- **Target:** BMI Value

## 🧠 Models Used
| Model | Type |
|-------|------|
| Linear Regression | Baseline model |
| Neural Network | 2 → 8 → 4 → 1 layers |

## ⚙️ Tech Stack
- Python
- NumPy / Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

## 📈 Results
| Metric | Linear Regression | Neural Network |
|--------|------------------|----------------|
| R² Score | ~0.99 | ~0.99 |
| MAE | ~0.10 | ~0.08 |
| Accuracy (±1 BMI) | ~98% | ~99% |

## 🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/bmi-prediction-ai.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model
python main.py
```

## 📁 Project Structure
