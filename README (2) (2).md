# Student Performance Prediction

Predict students' final scores and flag those at risk of falling behind, using machine learning on attendance, assessment and background data.

> **Important:** the bundled demo runs on a **synthetic dataset** of 1,000 students. The results show how the method works, not how it will perform on your students. Retrain on real, de-identified records before drawing any conclusions.

## What's in this project

| File | Purpose |
|---|---|
| `student_performance_prediction.py` | End-to-end pipeline: cleans data, trains models, prints metrics, saves charts |
| `streamlit_app.py` | Web app to predict one student or a whole class from a CSV |
| `Student_Performance_Prediction_Report.docx` | Full project report (7 pages) |
| Presentation deck | 11-slide summary of the project |

Files the script creates when you run it: `synthetic_students.csv` (demo data only), `plot_pred_vs_actual.png`, `plot_feature_importance.png`.

## What it does

1. **Predicts the final score** (regression) with four models: Linear Regression, Decision Tree, Random Forest and Gradient Boosting.
2. **Flags at-risk students** (classification): students whose final score is below a threshold (default 60), using Logistic Regression and Random Forest with class weighting.
3. **Explains the drivers** with permutation importance, showing which features matter most.

## Quick start

Requires Python 3.9 or newer.

```bash
pip install pandas numpy scikit-learn matplotlib
python student_performance_prediction.py
```

This runs on the synthetic demo data and prints the model comparison, feature importance and confusion matrices.

### Use your own data

```bash
python student_performance_prediction.py --csv students.csv --target final_score
```

| Option | Default | Meaning |
|---|---|---|
| `--csv` | none (demo data) | Path to your dataset |
| `--target` | `final_score` | Name of the numeric score column to predict |
| `--risk-threshold` | `60` | Scores below this count as "at risk" |

### Run the web app

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
streamlit run streamlit_app.py
```

The app has three tabs:
- **Predict one student:** enter details and get a predicted score, risk probability and status.
- **Predict a class (CSV):** upload a class file, get predictions sorted by risk, and download the results.
- **Model insights:** accuracy metrics and a feature importance chart.

The sidebar lets you upload your own training CSV, change the at-risk threshold, and switch between Linear Regression and Gradient Boosting.

## Data format

One row per student, one numeric target column, and any number of feature columns. The demo data uses:

| Column | Description |
|---|---|
| `study_hours` | Weekly study hours |
| `attendance_pct` | Class attendance (%) |
| `previous_score` | Score in the previous term (0 to 100) |
| `assignment_score` | Average continuous-assessment score (0 to 100) |
| `sleep_hours` | Average nightly sleep |
| `parental_education` | Highest parental education, coded 0 to 4 |
| `tutoring` | 1 if the student receives extra tutoring, else 0 |
| `absences` | Days absent |
| `final_score` | **Target:** final exam or course score (0 to 100) |

Cleaning is basic: duplicate rows are dropped, rows with a missing target are removed, missing numbers are filled with the median, missing text with `"unknown"`, and text columns are one-hot encoded. Review this against your own data. Real records usually need more careful handling of outliers and missing values.

## Results on the synthetic demo data

Split: 800 training and 200 test students, random seed 42.

| Model | MAE | RMSE | R² (test) | R² (5-fold CV) |
|---|---|---|---|---|
| Linear Regression | 3.98 | 5.02 | 0.789 | 0.792 |
| Gradient Boosting | 4.48 | 5.50 | 0.747 | 0.745 |
| Random Forest | 4.75 | 5.82 | 0.717 | 0.712 |
| Decision Tree | 6.29 | 7.94 | 0.473 | 0.516 |

At-risk detection (final score below 60, 48 at-risk students in the test set):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 84.5% | 63.5% | 83.3% | 0.721 |
| Random Forest | 86.0% | 68.5% | 77.1% | 0.725 |

Strongest predictors: previous score, study hours and assignment score.

Linear Regression comes out on top because the synthetic data was generated with mostly linear relationships. On real data, tree-based models may perform better.

## Limitations and responsible use

- **Synthetic data.** Nothing here describes real students.
- **Screening aid only.** A prediction should start a conversation with a student, never replace one. Typical error is about 4 marks, so treat scores as bands.
- **Fairness.** Features such as parental education can act as proxies for socioeconomic background. Check error rates across gender, language, income, caste and disability groups before real use, and consider dropping sensitive features.
- **Self-fulfilling labels.** An "at risk" flag can lower expectations. Keep flags confidential and frame them as an offer of support.
- **Privacy.** Student records are personal data. De-identify data used for modelling, limit access and follow your institution's data-protection policy.
- **Correlation is not causation.** Feature importance shows what predicts scores, not what causes them.

## Ideas for next steps

- Train on at least two past cohorts and hold out the most recent one as the test set.
- Add features such as assignment submission timing and subject-level scores.
- Tune hyperparameters and compare against the baselines above.
- Pilot with one class, measure outcomes against a comparison group, then scale.
- Refit every term and monitor accuracy over time.

## Reproducibility

Random seed 42 is used everywhere. Tested with Python 3.13, pandas 3.0, scikit-learn 1.9 and matplotlib. The Streamlit web page itself has not been run end to end in this environment; the model and prediction logic behind it was tested separately.
