import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load and clean data
df = pd.read_csv("loan.csv")
df.fillna(method='ffill', inplace=True)

# Drop Loan_ID if present
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Select only these 11 features for both training and UI
features = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area"]

X = df[features]
y = df["Loan_Status"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")