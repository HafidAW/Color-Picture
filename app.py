from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load colors.csv
colors_csv_path = os.path.join(BASE_DIR, 'colors.csv')
try:
    colors_df = pd.read_csv(colors_csv_path, names=['color', 'color_name', 'hex', 'R', 'G', 'B'])
except FileNotFoundError:
    print(f"Error: File '{colors_csv_path}' not found. Make sure the file exists in the correct location.")

# Load model
model_path = os.path.join(BASE_DIR, 'model_knn_colors.joblib')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found. Make sure the file exists in the correct location.")

# Route to serve HTML file
@app.route('/')
def index():
    html_path = os.path.join(BASE_DIR, 'index.html')
    return send_from_directory(os.path.dirname(html_path), os.path.basename(html_path))

@app.route('/colors', methods=['GET'])
def get_colors():
    if 'colors_df' in globals():  # Check if colors_df is loaded
        # Convert DataFrame to JSON and return
        colors_json = colors_df.to_json(orient='records')
        return jsonify(colors_json)
    else:
        return jsonify({'error': 'Failed to load colors data. Check server logs for details.'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'model' in globals():  # Check if model is loaded
        try:
            # Lakukan prediksi menggunakan model
            features = data['features']
            prediction = model.predict(features)
            # Format hasil prediksi menjadi JSON
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Failed to load model. Check server logs for details.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
