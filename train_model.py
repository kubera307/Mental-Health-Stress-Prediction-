print("🚀 Program started...\n")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("stress_detection_data.csv")
    print("✅ Dataset loaded successfully!\n")
except:
    print("❌ Error: CSV file not found!")
    exit()

print(df.head())
print("\nColumns:", df.columns)

# -----------------------------
# Preprocessing
# -----------------------------
le = LabelEncoder()
df['Stress_Detection'] = le.fit_transform(df['Stress_Detection'])

X = df[['Sleep_Duration', 'Work_Hours', 'Physical_Activity', 'Screen_Time']]
y = df['Stress_Detection']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")

# -----------------------------
# Evaluation
# -----------------------------
accuracy = model.score(X_test, y_test)
print("\n📊 Accuracy:", accuracy)

y_pred = model.predict(X_test)
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# Visualization
# -----------------------------
plt.scatter(df['Sleep_Duration'], df['Stress_Detection'])
plt.xlabel("Sleep Duration")
plt.ylabel("Stress Level")
plt.title("Sleep vs Stress")
plt.show()

# -----------------------------
# User Input Prediction
# -----------------------------
print("\n🔮 Enter your details:")

sleep = float(input("Sleep Hours: "))
work = float(input("Work Hours: "))
activity = float(input("Physical Activity: "))
screen = float(input("Screen Time: "))

sample = pd.DataFrame([[sleep, work, activity, screen]], columns=X.columns)

prediction = model.predict(sample)
result = le.inverse_transform(prediction)[0]

print("\n🧠 Predicted Stress Level:", result)

# -----------------------------
# Suggestions
# -----------------------------
if result == "High":
    print("⚠️ Suggestion: Reduce screen time, increase sleep, and do physical activity.")
elif result == "Medium":
    print("⚠️ Suggestion: Maintain a balanced routine.")
else:
    print("✅ Great! Keep maintaining your healthy lifestyle.")