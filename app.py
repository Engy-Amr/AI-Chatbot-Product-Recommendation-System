from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from DNCF import get_recommendations  # Import recommendation function from DNCF.py
import os
import pandas as pd
from keras.models import load_model

df = pd.read_csv('/Users/engyamr/Downloads/Python/output_dataset.csv')

model = load_model('/Users/engyamr/Downloads/Python/my_model.keras')
all_product_ids = df['Product ID'].unique()
product_names = dict(zip(df['Product ID'], df['Product']))

app = Flask(__name__, static_url_path='/static')

# Routes for serving static files (e.g., jQuery)
@app.route('/node_modules/<path:path>')
def send_node_modules(path):
    return send_from_directory(os.path.join('node_modules', 'jquery', 'dist'), path)

@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path, mimetype='application/javascript')

# Route for rendering the interface
@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Extract user_id from the request data
    user_id = request.form.get('user_id')

    # Perform recommendation based on user_id
    recommendations = get_recommendations(user_id, model, all_product_ids, df, product_names)

    # Check if recommendations is not empty
    if not recommendations.empty:
        # Convert DataFrame to list of dictionaries
        recommendations_dict = recommendations.to_dict(orient='records')
        return jsonify(recommendations_dict), 200
    else:
        return jsonify({'error': 'No recommendations found or an error occurred'}), 500

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)