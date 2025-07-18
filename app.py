from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

app = Flask(__name__)

# Train and save the model if it doesn't exist
if not os.path.exists('election_model.joblib'):
    from election_prediction import train_model, preprocess_data
    from data_generator import generate_sample_data
    
    # Generate sample data
    data = generate_sample_data()
    
    # Preprocess data
    processed_data = preprocess_data(data)
    X = processed_data.drop('outcome', axis=1)
    y = processed_data['outcome']
    
    # Train model
    model = train_model(X, y)
    
    # Save model and preprocessors
    joblib.dump(model, 'election_model.joblib')
    
    # Save label encoders
    label_encoders = {}
    categorical_cols = ['age_group', 'education_level', 'urban_rural', 'previous_voting']
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(data[col])
        label_encoders[col] = le
    joblib.dump(label_encoders, 'label_encoders.joblib')
    
    # Save scaler
    scaler = StandardScaler()
    scaler.fit(data[['income_level']])
    joblib.dump(scaler, 'scaler.joblib')

# Load the saved model and preprocessors
model = joblib.load('election_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    
    if request.method == 'POST':
        # Get form data
        input_data = {
            'age_group': request.form['age_group'],
            'income_level': float(request.form['income_level']),
            'education_level': request.form['education_level'],
            'urban_rural': request.form['urban_rural'],
            'previous_voting': request.form['previous_voting']
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess input data
        categorical_cols = ['age_group', 'education_level', 'urban_rural', 'previous_voting']
        for col in categorical_cols:
            input_df[col] = label_encoders[col].transform(input_df[col])
        
        input_df['income_level'] = scaler.transform(input_df[['income_level']])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
