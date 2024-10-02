from flask import Flask, request, jsonify
from flask_cors import CORS   
from material_recommender_model import recommend_material
from part_name_material import recommend_materials
from alloy import generate_alloy

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

    
# New endpoint for generating custom alloys
@app.route('/api/generate-alloy', methods=['POST'])
def generate_custom_alloy():
    data = request.json
    target_properties = data.get('target_properties', {})
    
    # Ensure all required properties are present
    required_properties = ['tensile_strength', 'hardness', 'thermal_resistance', 'density', 'sustainability_score']
    for prop in required_properties:
        if prop not in target_properties:
            return jsonify({'error': f'Missing required property: {prop}'}), 400

    try:
        new_composition, achieved_properties = generate_alloy(target_properties)
        
        # Calculate property match percentages
        property_matches = {}
        for prop in required_properties:
            match_percentage = (1 - abs(achieved_properties[prop] - target_properties[prop]) / target_properties[prop]) * 100
            property_matches[prop] = f'{match_percentage:.2f}%'

        response = {
            'composition': {element: f'{percentage:.2f}%' for element, percentage in new_composition.items()},
            'achieved_properties': {prop: f'{value:.2f}' for prop, value in achieved_properties.items()},
            'property_matches': property_matches
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)