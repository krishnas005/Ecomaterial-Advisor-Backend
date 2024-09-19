from flask import Flask, request, jsonify
from flask_cors import CORS   
from material_recommender_model import recommend_material
from part_name_material import recommend_materials

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Existing endpoint for general material recommendations based on user input
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input = data.get('properties', {})
    recommendations = recommend_material(user_input)
    return jsonify(recommendations.to_dict(orient='records'))

# New endpoint to return top 3 materials based on sustainability for a given part name
@app.route('/api/part-name', methods=['POST'])
def recommend_top_sustainable():
    data = request.json
    part_name = data.get('part_name', "")
    top_sustainable_recommendations = recommend_materials(part_name)
    return jsonify(top_sustainable_recommendations)

if __name__ == "__main__":
    app.run(debug=True)