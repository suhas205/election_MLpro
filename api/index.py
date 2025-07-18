import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__, template_folder="../templates")

# Load the saved model and preprocessors
model = joblib.load(os.path.join(os.path.dirname(__file__), '../election_model.joblib'))
label_encoders = joblib.load(os.path.join(os.path.dirname(__file__), '../label_encoders.joblib'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), '../scaler.joblib'))

# Read the HTML template
with open(os.path.join(os.path.dirname(__file__), '../templates/index.html'), 'r', encoding='utf-8') as f:
    INDEX_HTML = f.read()

@app.route('/', methods=['GET', 'POST'])
def handler():
    prediction = None
    if request.method == 'POST':
        input_data = {
            'age_group': request.form['age_group'],
            'income_level': float(request.form['income_level']),
            'education_level': request.form['education_level'],
            'urban_rural': request.form['urban_rural'],
            'previous_voting': request.form['previous_voting']
        }
        input_df = pd.DataFrame([input_data])
        categorical_cols = ['age_group', 'education_level', 'urban_rural', 'previous_voting']
        for col in categorical_cols:
            input_df[col] = label_encoders[col].transform(input_df[col])
        input_df['income_level'] = scaler.transform(input_df[['income_level']])
        prediction = model.predict(input_df)[0]
    return render_template_string(INDEX_HTML, prediction=prediction)

# For Vercel compatibility
handler = app
