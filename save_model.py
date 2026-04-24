# save_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the training data
print("Loading training data...")
df = pd.read_csv('Training.csv')

# Separate features and target
X = df.iloc[:, :-1]  # All columns except last
y = df.iloc[:, -1]   # Last column (prognosis)

print(f"Dataset shape: {X.shape}")
print(f"Number of unique diseases: {y.nunique()}")

# Encode the target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split for validation (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model and label encoder
joblib.dump(model, 'disease_model.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("\n✅ Files saved successfully!")
print(f"   - disease_model.joblib")
print(f"   - label_encoder.joblib")
print(f"\nNumber of features expected: {X.shape[1]}")
print(f"Number of disease classes: {len(le.classes_)}")
print(f"\nDiseases: {list(le.classes_)}")
