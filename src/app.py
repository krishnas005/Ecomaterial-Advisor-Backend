from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from material_recommender_model import recommend_material  # Ensure correct import path

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for testing purposes

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input = data.get('properties', {})
    recommendations = recommend_material(user_input)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True)
