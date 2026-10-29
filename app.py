from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ---------------------------------------------------------
# 1. Load Dataset & Train Model
# ---------------------------------------------------------
df = pd.read_csv("stress_detection_data.csv")

le_gender = LabelEncoder()
df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

le_stress = LabelEncoder()
df['Stress_Encoded'] = le_stress.fit_transform(df['Stress_Detection'])

features = [
    'Age',
    'Gender_Encoded',
    'Sleep_Duration',
    'Work_Hours',
    'Physical_Activity',
    'Screen_Time'
]

X = df[features]
y = df['Stress_Encoded']

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# ---------------------------------------------------------
# 2. Suggestion System
# ---------------------------------------------------------
def get_suggestion(stress_level):
    stress_level = stress_level.lower()

    if "high" in stress_level:
        return "⚠️ High Stress: Reduce workload, sleep 7–8 hrs, practice meditation."
    elif "medium" in stress_level:
        return "⚡ Medium Stress: Maintain balance between work and rest."
    else:
        return "✅ Low Stress: Great! Keep your healthy lifestyle."

# ---------------------------------------------------------
# 3. Routes
# ---------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ---------------- Input from form ----------------
        age = float(request.form.get('age'))
        gender_raw = request.form.get('gender')
        sleep = float(request.form.get('sleep'))
        work = float(request.form.get('work'))
        activity = float(request.form.get('activity'))
        screen = float(request.form.get('screen'))

        gender_encoded = 1 if gender_raw == "Male" else 0

        sample = pd.DataFrame([[age, gender_encoded, sleep, work, activity, screen]], columns=features)

        # ---------------- ML Prediction ----------------
        prediction_idx = model.predict(sample)[0]
        prediction_text = le_stress.inverse_transform([prediction_idx])[0]

        # ------------------------------------------------
        # REAL STRESS SCORE (MODEL + LIFESTYLE FACTORS)
        # ------------------------------------------------

        base_prob = model.predict_proba(sample)[0][prediction_idx] * 100

        risk_score = 0

        # Sleep impact
        if sleep < 5:
            risk_score += 25
        elif sleep < 7:
            risk_score += 10

        # Work hours impact
        if work > 8:
            risk_score += 25
        elif work > 6:
            risk_score += 10

        # Screen time impact
        if screen > 6:
            risk_score += 15

        # Physical activity impact
        if activity < 1:
            risk_score += 10

        # Final stress score
        final_score = round((base_prob * 0.6) + (risk_score * 0.4), 2)
        final_score = min(100, final_score)

        score_display = f"{final_score}% Stress Level"

        # ---------------- Suggestion ----------------
        suggestion = get_suggestion(prediction_text)

        return render_template(
            'index.html',
            prediction=prediction_text,
            score=score_display,
            suggestion=suggestion
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction="Error",
            score=str(e),
            suggestion=""
        )


# ---------------------------------------------------------
# 4. Run App
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)