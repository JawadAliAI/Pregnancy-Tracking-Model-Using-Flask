from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load models and scalers
import joblib
import numpy as np

gbr_model = joblib.load('gbr_regression_model.pkl')
xgb_model = joblib.load('xgboost_classification_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load LabelEncoders
risk_label_encoder = joblib.load('risk_label_encoder.pkl')
water_label_encoder = joblib.load('water_label_encoder.pkl')


@app.route('/')
def home():
    # Serve the main index.html page
    return render_template('index.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Add a custom 404 error page
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404-error.html'), 404



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the form
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi = float(request.form['bmi'])
        gestational_week = float(request.form['gestational_week'])
        contractions = float(request.form['contractions'])
        mood_changes = float(request.form['mood_changes'])
        symptoms = float(request.form['symptoms'])
        sleep_patterns = float(request.form['sleep_patterns'])
        dietary_habits = float(request.form['dietary_habits'])
        stress_level_low = int(request.form.get('stress_level_low', 0))
        stress_level_moderate = int(request.form.get('stress_level_moderate', 0))
        amniotic_fluid_low = int(request.form.get('amniotic_fluid_low', 0))
        amniotic_fluid_normal = int(request.form.get('amniotic_fluid_normal', 0))
        preferred_notification_sms = int(request.form.get('preferred_notification_sms', 0))
        fatigue_low = int(request.form.get('fatigue_low', 0))
        fatigue_moderate = int(request.form.get('fatigue_moderate', 0))
        days_since_lmp = int(request.form['days_since_lmp'])
        weeks_of_pregnancy = int(request.form['weeks_of_pregnancy'])
        total_pregnancy_duration = int(request.form['total_pregnancy_duration'])

        # Prepare features for the model
        features = [
            age, weight, height, bmi, gestational_week, contractions, mood_changes,
            symptoms, sleep_patterns, dietary_habits, stress_level_low, stress_level_moderate,
            amniotic_fluid_low, amniotic_fluid_normal, preferred_notification_sms, fatigue_low,
            fatigue_moderate, days_since_lmp, weeks_of_pregnancy, total_pregnancy_duration
        ]
        scaled_features = scaler.transform([features])

        # Make predictions (Regression)
        reg_prediction = gbr_model.predict(scaled_features)

        # Make predictions (Classification)
        class_prediction = xgb_model.predict(scaled_features)

        # Decode classification predictions
        risk_prediction_label = risk_label_encoder.inverse_transform([class_prediction[0][0]])
        water_prediction_label = water_label_encoder.inverse_transform([class_prediction[0][1]])

        # Prepare response
        response = {
            'predicted_weeks_of_pregnancy': reg_prediction[0][0],
            'predicted_additional_metric': reg_prediction[0][1] if len(reg_prediction[0]) > 1 else None,
            'risk_assessment': risk_prediction_label[0],
            'water_breakage_likelihood': water_prediction_label[0]
        }

        # Return JSON response
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
