from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============ LOAD MODEL ============
try:
    model = joblib.load('cbam_model.pkl')
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None


# ============ FEATURE ENGINEERING ============
def engineer_features(df):
    df = df.copy()
    df['emission_intensity'] = df['embedded_emissions_tco2'] / df['quantity_tonnes'].replace(0, 1)
    df['emission_intensity'] = df['emission_intensity'].fillna(0)

    df['carbon_price_gap'] = df['eu_ets_price_eur'] - df['carbon_price_origin_eur']
    df['total_emissions'] = df['direct_emissions_tco2'] + df['indirect_emissions_tco2']
    df['cost_per_tonne'] = 0
    df['emission_ratio'] = df['direct_emissions_tco2'] / (df['total_emissions'] + 1)
    df['price_ratio'] = df['carbon_price_origin_eur'] / (df['eu_ets_price_eur'] + 1)
    df['emission_to_quantity'] = df['total_emissions'] / df['quantity_tonnes'].replace(0, 1)
    df['high_emission_flag'] = 0
    df['high_price_gap_flag'] = 0
    df['log_quantity'] = np.log1p(df['quantity_tonnes'])
    df['log_emissions'] = np.log1p(df['total_emissions'])

    return df


# ============ ROUTES ============

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'running', 'model': 'CBAM API v1.0'}), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data'}), 400

        df_input = pd.DataFrame([data])
        df_input = engineer_features(df_input)

        categorical = model_info['categorical']
        numerical = model_info['numerical']
        required_cols = categorical + numerical

        prediction = model.predict(df_input[required_cols])[0]

        return jsonify({
            'prediction': float(prediction),
            'unit': 'EUR',
            'status': 'success'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        payload = request.get_json()

        if not payload:
            return jsonify({'error': 'No data received'}), 400

        # Handle both dict and list
        if isinstance(payload, dict) and 'data' in payload:
            data_list = payload['data']
        elif isinstance(payload, list):
            data_list = payload
        else:
            return jsonify({'error': 'Invalid format. Send {"data": [...]} or [...]'}), 400

        if not data_list:
            return jsonify({'error': 'Empty data array'}), 400

        df_input = pd.DataFrame(data_list)
        df_input = engineer_features(df_input)

        required_cols = model_info['categorical'] + model_info['numerical']
        predictions = model.predict(df_input[required_cols])

        return jsonify({
            'total_records': len(predictions),
            'predictions': [{'record': i + 1, 'prediction': float(p)} for i, p in enumerate(predictions)],
            'status': 'success'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/test', methods=['GET', 'POST'])
def test():
    return jsonify({'message': 'API is working!', 'methods': ['GET', 'POST']}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)