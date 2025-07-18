import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def preprocess_data(data):
    """Preprocess the election data."""
    # Create copy of data
    df = data.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['age_group', 'education_level', 'urban_rural', 'previous_voting']
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Scale numerical variables
    scaler = StandardScaler()
    df['income_level'] = scaler.fit_transform(df[['income_level']])
    
    return df

def train_model(X_train, y_train):
    """Train the election prediction model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    predictions = model.predict(X_test)
    
    print("\nModel Evaluation:")
    print("----------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importance = abs(model.coef_[0])
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance)
    plt.title('Feature Importance in Election Prediction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Load data (generate if it doesn't exist)
    try:
        data = pd.read_csv('election_data.csv')
    except FileNotFoundError:
        from data_generator import generate_sample_data
        data = generate_sample_data()
        data.to_csv('election_data.csv', index=False)
    
    # Preprocess data
    processed_data = preprocess_data(data)
    
    # Separate features and target
    X = processed_data.drop('outcome', axis=1)
    y = processed_data['outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(model, X.columns)
    
    print("\nFeature importance plot saved as 'feature_importance.png'")

if __name__ == "__main__":
    main()
