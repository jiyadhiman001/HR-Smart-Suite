# 🧑‍💼 HR Smart Suite

An AI-powered HR analytics toolkit that helps HR teams make smarter, data-driven decisions.

---

## 🚀 Live Demo

👉 [HR Smart Suite on Streamlit Cloud](https://hr-smart-suite.streamlit.app)

---

## 📌 Features

### Phase 1 (Current)
| Feature | Description |
|---|---|
| 📉 Employee Attrition Predictor | Predicts whether an employee is at risk of leaving, based on satisfaction, income, overtime, tenure, and more |
| ✅ Hiring Success Predictor | Predicts whether a candidate is likely to succeed in a role, based on experience, skills, interview, and test scores |
| 📂 CSV Bulk Upload | HR can upload their own company data to get predictions for multiple employees/candidates at once |
| 🤖 Groq AI Insights | After every prediction, an AI HR consultant provides a personalized action plan |

### Phase 2 (Roadmap)
- 🔥 Employee Burnout / Absenteeism Predictor
- 📈 Performance Predictor
- 💰 Salary Hike / Promotion Eligibility Predictor

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| ML Models | scikit-learn (Random Forest), XGBoost |
| Class Imbalance Handling | SMOTE (imbalanced-learn) |
| AI Insights | Groq API (llama-3.3-70b-versatile) |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |

---

## 📊 Models

### Employee Attrition Predictor
- **Dataset**: IBM HR Analytics Employee Attrition Dataset (1,470 employees, 35 features)
- **Algorithm**: XGBoost with `scale_pos_weight` to handle class imbalance
- **Performance**: 85% Accuracy, 0.78 ROC-AUC

### Hiring Success Predictor
- **Dataset**: Synthetic candidate dataset (1,500 candidates, 7 features)
- **Algorithm**: Random Forest Classifier
- **Performance**: 77% Accuracy, 0.80 ROC-AUC, 87% Recall

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/jiyadhiman001/HR-Smart-Suite.git
cd HR-Smart-Suite

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the models
python generate_hiring_data.py
python train_attrition_final.py
python train_hiring_model.py

# 5. Run the app
streamlit run app.py
```

---

## ⚠️ Disclaimer

This tool is designed to assist HR decision-making, not replace it. All final decisions must be made by qualified HR professionals.

---

## 👩‍💻 Author

**Jiya Dhiman**
B.Tech CSE | Hindu College of Engineering, Sonipat

[![GitHub](https://img.shields.io/badge/GitHub-jiyadhiman001-black?logo=github)](https://github.com/jiyadhiman001)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-blue)](https://jiyadhiman001.github.io/jiyadhiman_portfolio/)