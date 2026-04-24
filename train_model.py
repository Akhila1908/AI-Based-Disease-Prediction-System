# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os

def train_and_save():
    print("Loading training data...")
    # Load the CSV from your repository
    df = pd.read_csv('Training.csv')
    
    # Drop unnamed column if exists
    df = df.drop(columns=['Unnamed: 133'], errors='ignore')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}...")
    
    # Separate features and target
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    print(f"Number of disease classes: {len(y.unique())}")
    
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Create pipeline (without scaler - Random Forest doesn't need it)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Training model...")
    pipeline.fit(X, y_encoded)
    
    # Save with compatibility settings
    print("Saving model...")
    joblib.dump(pipeline, 'disease_model.joblib', compress=3)
    joblib.dump(le, 'label_encoder.joblib', compress=3)
    
    # Also save as pickle as backup
    import pickle
    with open('disease_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f, protocol=4)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f, protocol=4)
    
    print("✅ Models saved successfully!")
    print(f"Model classes: {le.classes_[:5]}...")
    
    # Test prediction
    test_input = np.zeros((1, X.shape[1]))
    test_pred = pipeline.predict(test_input)
    test_disease = le.inverse_transform(test_pred)
    print(f"Test prediction on zero input: {test_disease[0]}")
    
    return pipeline, le

if __name__ == "__main__":
    train_and_save()
