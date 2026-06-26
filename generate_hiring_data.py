import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

experience_years = np.round(np.random.exponential(scale=3, size=N), 1)
experience_years = np.clip(experience_years, 0, 20)

education_level = np.random.choice(
    ["High School", "Bachelor's", "Master's", "PhD"],
    size=N, p=[0.10, 0.55, 0.30, 0.05]
)
education_score_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
education_score = np.array([education_score_map[e] for e in education_level])

skill_match_pct = np.clip(np.random.normal(65, 18, N), 10, 100).round(1)
interview_score = np.clip(np.random.normal(6.2, 1.8, N), 1, 10).round(1)
technical_test_score = np.clip(np.random.normal(62, 20, N), 0, 100).round(1)
communication_score = np.clip(np.random.normal(6.5, 1.5, N), 1, 10).round(1)
prev_job_stability = np.clip(np.random.normal(2.2, 1.1, N), 0.2, 8).round(1)

score = (
    0.18 * (experience_years / 20) +
    0.12 * (education_score / 4) +
    0.20 * (skill_match_pct / 100) +
    0.20 * (interview_score / 10) +
    0.15 * (technical_test_score / 100) +
    0.10 * (communication_score / 10) +
    0.05 * (prev_job_stability / 8)
)

noise = np.random.normal(0, 0.08, N)
final_score = score + noise
threshold = np.percentile(final_score, 38)
hired_success = (final_score > threshold).astype(int)

df = pd.DataFrame({
    "ExperienceYears": experience_years,
    "EducationLevel": education_level,
    "SkillMatchPercent": skill_match_pct,
    "InterviewScore": interview_score,
    "TechnicalTestScore": technical_test_score,
    "CommunicationScore": communication_score,
    "PrevJobStabilityYears": prev_job_stability,
    "HiringSuccess": hired_success
})

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/hiring_success.csv", index=False)
print("Done! Shape:", df.shape)