import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq

st.set_page_config(page_title="HR Smart Suite", page_icon="🧑‍💼", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ─── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}

.hero {
    background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.85);
    margin: 0.5rem 0 0 0;
}

.result-risk {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-prob {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.risk-color { color: #f87171; }
.safe-color { color: #34d399; }

.ai-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(14,165,233,0.1));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.ai-box h4 {
    color: #a5b4fc;
    margin-top: 0;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #38bdf8;
    margin-bottom: 0.5rem;
    background: rgba(14,165,233,0.15);
    padding: 4px 10px;
    border-radius: 4px;
    display: inline-block;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    color: white !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    color: white !important;
    width: 100%;
}

[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

hr { border-color: rgba(255,255,255,0.1) !important; }
.stAlert { border-radius: 10px !important; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 1rem;
}
/* Fix input labels */
label, .stNumberInput label, .stSelectbox label, 
.stSlider label, .stTextInput label {
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* Fix metric text */
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.5rem !important;
}
/* Fix radio buttons */
.stRadio label p {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}
.stRadio > label {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load models ──────────────────────────────────────────
@st.cache_resource
def load_attrition_artifacts():
    model = joblib.load(f"{MODEL_DIR}/attrition_model.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/attrition_encoders.pkl")
    features = joblib.load(f"{MODEL_DIR}/attrition_features.pkl")
    threshold = joblib.load(f"{MODEL_DIR}/attrition_threshold.pkl")
    return model, encoders, features, threshold

@st.cache_resource
def load_hiring_artifacts():
    model = joblib.load(f"{MODEL_DIR}/hiring_model.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/hiring_encoders.pkl")
    features = joblib.load(f"{MODEL_DIR}/hiring_features.pkl")
    return model, encoders, features

attrition_model, attrition_encoders, attrition_features, attrition_threshold = load_attrition_artifacts()
hiring_model, hiring_encoders, hiring_features = load_hiring_artifacts()

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧑‍💼 HR Smart Suite")
    st.markdown("*AI-powered HR analytics toolkit*")
    st.divider()
    st.markdown("**Phase 1**")
    st.markdown("- 📉 Employee Attrition Predictor\n- ✅ Hiring Success Predictor")
    st.markdown("**Coming Soon**")
    st.markdown("- 🔥 Burnout Risk Predictor\n- 📈 Performance Predictor\n- 💰 Promotion Eligibility")
    st.divider()
    st.markdown("**🤖 Groq AI Insights**")
    groq_api_key = st.text_input("Groq API Key (optional)", type="password", help="Get free key at console.groq.com")
    st.divider()
    st.caption("DevAlpha Foundation · AI Internship 2026\nTask 3: Prediction System")

# ─── Hero ─────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>HR Smart Suite 🧑‍💼</h1>
    <p>AI-powered predictions to help HR teams make smarter, faster, and fairer decisions</p>
</div>
""", unsafe_allow_html=True)

# ─── Groq helpers ─────────────────────────────────────────
def get_groq_attrition_insights(emp, prob, at_risk):
    try:
        client = Groq(api_key=groq_api_key)
        prompt = f"""You are an expert HR consultant analyzing employee attrition risk.

Employee: Age {emp['Age']}, Income {emp['MonthlyIncome']}, Dept {emp['Department']},
Job Satisfaction {emp['JobSatisfaction']}/4, Work-Life Balance {emp['WorkLifeBalance']}/4,
OverTime: {emp['OverTime']}, Years at Company: {emp['YearsAtCompany']},
Stock Options: {emp['StockOptionLevel']}/3, Marital: {emp['MaritalStatus']}

Attrition Risk: {"HIGH" if at_risk else "LOW"} ({prob*100:.1f}%)

Give a concise HR action plan:
1. **Key Risk Factors** (2-3 bullets)
2. **Recommended Actions** (2-3 bullets)
3. **Retention Strategy** (1-2 sentences)

Be professional and actionable. No need to repeat the data."""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.5
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ Could not fetch insights: {e}"

def get_groq_hiring_insights(cand, prob, success):
    try:
        client = Groq(api_key=groq_api_key)
        prompt = f"""You are an expert HR recruiter analyzing hiring success potential.

Candidate: {cand['ExperienceYears']}yrs exp, {cand['EducationLevel']},
Skill Match {cand['SkillMatchPercent']}%, Interview {cand['InterviewScore']}/10,
Technical {cand['TechnicalTestScore']}/100, Communication {cand['CommunicationScore']}/10,
Job Stability {cand['PrevJobStabilityYears']} yrs/job avg

Predicted: {"LIKELY TO SUCCEED" if success else "LOWER SUCCESS LIKELIHOOD"} ({prob*100:.1f}%)

Give a concise recruiter report:
1. **Strengths** (2 bullets)
2. **Concerns** (2 bullets)
3. **Hiring Recommendation** (1-2 sentences)"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.5
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ Could not fetch insights: {e}"

# ─── Tabs ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📉  Employee Attrition Predictor", "✅  Hiring Success Predictor"])

# ══════════════════════════════════════════════════════════
# TAB 1 — ATTRITION
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Attrition Risk Analysis</p>', unsafe_allow_html=True)
    st.markdown("Identify employees who may be at risk of leaving — so HR can act before it's too late.")

    mode = st.radio("Input mode", ["Single Employee", "Bulk CSV Upload"], horizontal=True, key="att_mode")

    if mode == "Single Employee":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p class="section-label">Personal Info</p>', unsafe_allow_html=True)
            age = st.number_input("Age", 18, 60, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance_from_home = st.number_input("Distance From Home (km)", 0, 50, 10)
            num_companies_worked = st.number_input("Companies Worked At", 0, 15, 2)

        with col2:
            st.markdown('<p class="section-label">Job Details</p>', unsafe_allow_html=True)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            overtime = st.selectbox("OverTime", ["No", "Yes"])
            monthly_income = st.number_input("Monthly Income (₹/$)", 1000, 200000, 30000, step=1000)

        with col3:
            st.markdown('<p class="section-label">Satisfaction & Tenure</p>', unsafe_allow_html=True)
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3, help="1=Low, 4=Very High")
            environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
            work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
            job_involvement = st.slider("Job Involvement", 1, 4, 3)
            stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
            years_at_company = st.number_input("Years At Company", 0, 40, 3)
            total_working_years = st.number_input("Total Working Years", 0, 40, 5)
            years_with_curr_manager = st.number_input("Years With Manager", 0, 20, 2)

        if st.button("🔍 Predict Attrition Risk", type="primary"):
            defaults = {
                "Age": age, "BusinessTravel": business_travel, "DailyRate": 800,
                "Department": department, "DistanceFromHome": distance_from_home,
                "Education": 3, "EducationField": "Life Sciences",
                "EnvironmentSatisfaction": environment_satisfaction, "Gender": gender,
                "HourlyRate": 65, "JobInvolvement": job_involvement, "JobLevel": 2,
                "JobRole": job_role, "JobSatisfaction": job_satisfaction,
                "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
                "MonthlyRate": 14000, "NumCompaniesWorked": num_companies_worked,
                "OverTime": overtime, "PercentSalaryHike": 14, "PerformanceRating": 3,
                "RelationshipSatisfaction": 3, "StockOptionLevel": stock_option_level,
                "TotalWorkingYears": total_working_years, "TrainingTimesLastYear": 2,
                "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": min(years_with_curr_manager, years_at_company),
                "YearsSinceLastPromotion": 1, "YearsWithCurrManager": years_with_curr_manager,
            }
            row = pd.DataFrame([defaults])[attrition_features]
            for col, enc in attrition_encoders.items():
                row[col] = enc.transform(row[col].astype(str))

            proba = attrition_model.predict_proba(row)[0][1]
            prediction = int(proba >= attrition_threshold)

            st.divider()
            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                if prediction == 1:
                    st.markdown(f"""<div class="result-risk">
                        <div style="font-size:2rem">⚠️</div>
                        <div class="result-title risk-color">At Risk</div>
                        <div class="result-prob risk-color">{proba*100:.1f}%</div>
                        <div style="color:#94a3b8;font-size:0.85rem">Attrition Probability</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-safe">
                        <div style="font-size:2rem">✅</div>
                        <div class="result-title safe-color">Likely to Stay</div>
                        <div class="result-prob safe-color">{proba*100:.1f}%</div>
                        <div style="color:#94a3b8;font-size:0.85rem">Attrition Probability</div>
                    </div>""", unsafe_allow_html=True)
            with r2:
                st.metric("Stay Probability", f"{(1-proba)*100:.1f}%")
                st.metric("Risk Level", "High 🔴" if prediction == 1 else "Low 🟢")
            with r3:
                st.info("This is a risk **indicator**, not a verdict. Use it as a signal to have a meaningful conversation with the employee — HR judgment always comes first.")

            if groq_api_key:
                st.divider()
                with st.spinner("🤖 Generating HR insights..."):
                    insights = get_groq_attrition_insights(defaults, proba, bool(prediction))
                st.markdown('<div class="ai-box"><h4>🤖 AI HR Insights</h4>' + insights.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ai-box"><h4>🤖 AI HR Insights</h4><p style="color:#94a3b8">Add your Groq API key in the sidebar to get personalized HR action plans after every prediction.</p></div>', unsafe_allow_html=True)

    else:
        st.info(f"Upload employee CSV with columns: `{', '.join(attrition_features)}`")
        uploaded = st.file_uploader("Upload CSV", type="csv", key="att_csv")
        if uploaded:
            try:
                user_df = pd.read_csv(uploaded)
                missing = [c for c in attrition_features if c not in user_df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    proc = user_df[attrition_features].copy()
                    for col, enc in attrition_encoders.items():
                        proc[col] = proc[col].astype(str).apply(lambda x: x if x in set(enc.classes_) else enc.classes_[0])
                        proc[col] = enc.transform(proc[col])
                    probs = attrition_model.predict_proba(proc)[:, 1]
                    preds = (probs >= attrition_threshold).astype(int)
                    result = user_df.copy()
                    result["Attrition_Risk_%"] = (probs * 100).round(1)
                    result["Prediction"] = np.where(preds == 1, "⚠️ At Risk", "✅ Likely to Stay")
                    st.success(f"✅ Processed {len(result)} employees")
                    st.dataframe(result, use_container_width=True)
                    st.download_button("⬇️ Download Results", result.to_csv(index=False).encode(), "attrition_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════
# TAB 2 — HIRING
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Hiring Success Analysis</p>', unsafe_allow_html=True)
    st.markdown("Evaluate whether a candidate is likely to thrive in the role before making the hire.")

    mode2 = st.radio("Input mode", ["Single Candidate", "Bulk CSV Upload"], horizontal=True, key="hire_mode")

    if mode2 == "Single Candidate":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-label">Background</p>', unsafe_allow_html=True)
            experience = st.number_input("Experience (Years)", 0.0, 30.0, 2.0, step=0.5)
            education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
            skill_match = st.slider("Skill Match (%)", 0, 100, 65)
            prev_stability = st.number_input("Avg. Years per Previous Job", 0.0, 15.0, 2.0, step=0.5)

        with col2:
            st.markdown('<p class="section-label">Assessment Scores</p>', unsafe_allow_html=True)
            interview_score = st.slider("Interview Score", 0.0, 10.0, 6.0, step=0.1)
            technical_score = st.slider("Technical Test Score", 0, 100, 60)
            communication_score = st.slider("Communication Score", 0.0, 10.0, 6.5, step=0.1)

        if st.button("🔍 Predict Hiring Success", type="primary"):
            cand = {
                "ExperienceYears": experience, "EducationLevel": education,
                "SkillMatchPercent": skill_match, "InterviewScore": interview_score,
                "TechnicalTestScore": technical_score, "CommunicationScore": communication_score,
                "PrevJobStabilityYears": prev_stability
            }
            row = pd.DataFrame([cand])[hiring_features]
            for col, enc in hiring_encoders.items():
                row[col] = enc.transform(row[col].astype(str))

            proba = hiring_model.predict_proba(row)[0][1]
            prediction = int(proba >= 0.5)

            st.divider()
            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                if prediction == 1:
                    st.markdown(f"""<div class="result-safe">
                        <div style="font-size:2rem">🌟</div>
                        <div class="result-title safe-color">Strong Fit</div>
                        <div class="result-prob safe-color">{proba*100:.1f}%</div>
                        <div style="color:#94a3b8;font-size:0.85rem">Success Probability</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-risk">
                        <div style="font-size:2rem">⚠️</div>
                        <div class="result-title risk-color">Needs Review</div>
                        <div class="result-prob risk-color">{proba*100:.1f}%</div>
                        <div style="color:#94a3b8;font-size:0.85rem">Success Probability</div>
                    </div>""", unsafe_allow_html=True)
            with r2:
                st.metric("Success Probability", f"{proba*100:.1f}%")
                st.metric("Fit Level", "Strong 🟢" if prediction == 1 else "Moderate 🟡")
            with r3:
                st.info("This score reflects patterns from historical hiring data. Always consider cultural fit, team dynamics, and growth potential alongside this prediction.")

            if groq_api_key:
                st.divider()
                with st.spinner("🤖 Generating recruiter insights..."):
                    insights2 = get_groq_hiring_insights(cand, proba, bool(prediction))
                st.markdown('<div class="ai-box"><h4>🤖 AI Recruiter Insights</h4>' + insights2.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ai-box"><h4>🤖 AI Recruiter Insights</h4><p style="color:#94a3b8">Add your Groq API key in the sidebar to get AI-powered recruiter recommendations.</p></div>', unsafe_allow_html=True)

    else:
        st.info(f"Upload candidate CSV with columns: `{', '.join(hiring_features)}`")
        uploaded2 = st.file_uploader("Upload CSV", type="csv", key="hire_csv")
        if uploaded2:
            try:
                user_df2 = pd.read_csv(uploaded2)
                missing2 = [c for c in hiring_features if c not in user_df2.columns]
                if missing2:
                    st.error(f"Missing columns: {missing2}")
                else:
                    proc2 = user_df2[hiring_features].copy()
                    for col, enc in hiring_encoders.items():
                        proc2[col] = proc2[col].astype(str).apply(lambda x: x if x in set(enc.classes_) else enc.classes_[0])
                        proc2[col] = enc.transform(proc2[col])
                    probs2 = hiring_model.predict_proba(proc2)[:, 1]
                    preds2 = (probs2 >= 0.5).astype(int)
                    result2 = user_df2.copy()
                    result2["Success_%"] = (probs2 * 100).round(1)
                    result2["Prediction"] = np.where(preds2 == 1, "🌟 Strong Fit", "⚠️ Needs Review")
                    st.success(f"✅ Processed {len(result2)} candidates")
                    st.dataframe(result2, use_container_width=True)
                    st.download_button("⬇️ Download Results", result2.to_csv(index=False).encode(), "hiring_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("HR Smart Suite · Built by Jiya Dhiman")