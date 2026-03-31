import pandas as pd
import numpy as np
import os
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

# =========================
# Load dataset
# =========================
file_path = "student_dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file '{file_path}' not found!")

df = pd.read_csv(file_path).copy()

# =========================
# Validate required columns
# =========================
required_columns = ['Age', 'Gender', 'Country', 'State', 'City', 'Parent Occupation',
                    'Earning Class', 'Level of Student', 'Level of Course',
                    'Course Name', 'Study Time Per Day', 'IQ of Student', 'Assessment Score']

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# =========================
# Encode categorical columns
# =========================
label_encoders = {}
categorical_columns = ['Gender', 'Country', 'State', 'City',
                       'Parent Occupation', 'Earning Class',
                       'Level of Student', 'Level of Course', 'Course Name']

for col in categorical_columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# =========================
# Feature selection
# =========================
X = df[required_columns[:-1]]
y = df['Assessment Score']

# =========================
# Standardization
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# Model training
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel MAE: {mae:.2f}")

# =========================
# Predictions
# =========================
df['Predicted Score'] = model.predict(X_scaled)
df['Promotion Status'] = df['Predicted Score'].apply(
    lambda x: 'Promoted' if x >= 50 else 'Not Promoted'
)

# =========================
# Study material recommendation
# =========================
def filter_material(level):
    if pd.isna(level):
        return "Unknown"
    elif level <= 1:
        return "Basic Materials"
    elif level <= 3:
        return "Intermediate Materials"
    else:
        return "Advanced Materials"

df['Recommended Material'] = df['Level of Student'].apply(filter_material)

# =========================
# Save results
# =========================
output_file_path = "Final_predictions.xlsx"
df.to_excel(output_file_path, index=False)

# =========================
# Display results
# =========================
print("\n========== Student Assessment Results ==========")
print(df[['Name', 'Assessment Score', 'Predicted Score',
          'Promotion Status', 'Recommended Material']].head().to_string(index=False))
print("===============================================")
print(f"\nResults saved to: {output_file_path}\n")

# =========================
# Learning Material Provider
# =========================
def provide_material():
    default_pdf = "M1_Data Warehousing.pdf"
    explained_pdf = "M2_Data Warehousing.pdf"
    default_video = "https://youtu.be/gmvvaobm7eQ"
    explained_video = "https://youtu.be/J_LnPL3Qg70"

    print("\n📚 Opening recommended learning materials...")

    if os.path.exists(default_pdf):
        webbrowser.open(default_pdf)
    else:
        print(f"[Warning] {default_pdf} not found.")

    print(f"▶ Intro Video: {default_video}")
    webbrowser.open(default_video)

    user_input = input("\nNeed detailed materials? (Y/N): ").strip().lower()

    if user_input in ["y", "yes"]:
        if os.path.exists(explained_pdf):
            webbrowser.open(explained_pdf)
        else:
            print(f"[Warning] {explained_pdf} not found.")

        print(f"▶ Detailed Video: {explained_video}")
        webbrowser.open(explained_video)
    else:
        print("\n👍 Okay! You can explore more anytime.")

# Run
provide_material()
