import pandas as pd
import numpy as np
import os
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

# Load dataset
file_path = "/content/student_dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file '{file_path}' not found. Please ensure the file is available.")

df = pd.read_csv(file_path)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Country', 'State', 'City', 'Parent Occupation', 'Earning Class',
                       'Level of Student', 'Level of Course', 'Course Name']

for col in categorical_columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Feature selection
X = df[['Age', 'Gender', 'Country', 'State', 'City', 'Parent Occupation', 'Earning Class',
        'Level of Student', 'Level of Course', 'Course Name', 'Study Time Per Day', 'IQ of Student']]
y = df['Assessment Score']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df['Predicted Score'] = model.predict(X)
df['Promotion Status'] = df['Predicted Score'].apply(lambda x: 'Promoted' if x >= 50 else 'Not Promoted')

# Function to recommend study material
def filter_material(level):
    if pd.isna(level):
        return "Unknown"
    elif level < 0:
        return "Basic Materials"
    elif 1 <= level <= 2:
        return "Intermediate Materials"
    else:
        return "Advanced Materials"

df['Recommended Material'] = df['Level of Student'].apply(lambda x: filter_material(x))

# Save results
output_file_path = "Final_predictions.xlsx"
df.to_excel(output_file_path, index=False)

# Display formatted results
print("\n========== Student Assessment Results ==========")
print(df[['Name', 'Assessment Score', 'Predicted Score', 'Promotion Status', 'Recommended Material']].head().to_string(index=False))
print("\n===============================================")
print(f"\nFull results have been saved to: {output_file_path}\n")

# Provide study material
def provide_material():
    default_pdf = "M1_Data Warehousing.pdf"
    explained_pdf = "M2_Data Warehousing.pdf"
    default_video = "https://youtu.be/gmvvaobm7eQ"
    explained_video = "https://youtu.be/J_LnPL3Qg70"

    print("\nOpening recommended learning materials...")
    
    if os.path.exists(default_pdf):
        print(f"Opening: {default_pdf}")
        webbrowser.open(default_pdf)
    else:
        print(f"[Warning] Default PDF '{default_pdf}' not found.")
    
    print(f"Watch the introductory video here: {default_video}")
    webbrowser.open(default_video)

    while True:
        user_input = input("\nWould you like a detailed explanation with additional materials? (Yes/No): ").strip().lower()
        if user_input in ["yes", "y"]:
            if os.path.exists(explained_pdf):
                print(f"Opening: {explained_pdf}")
                webbrowser.open(explained_pdf)
            else:
                print(f"[Warning] Detailed PDF '{explained_pdf}' not found.")
            print(f"Watch the detailed video here: {explained_video}")
            webbrowser.open(explained_video)
            break
        elif user_input in ["no", "n"]:
            print("\n[Info] No additional materials will be provided. Feel free to revisit if needed!")
            break
        else:
            print("[Error] Invalid input. Please enter 'Yes' or 'No' (or 'Y'/'N').")

provide_material()
