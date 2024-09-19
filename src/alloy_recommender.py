import json
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global variable to store materials data
all_materials = {}

# Load materials data when the app starts
def load_materials():
    global all_materials
    file_path = r"../data/materials.json"  
    with open(file_path, 'r') as f:
        all_materials = json.load(f)

# Normalize data for comparison
def normalize(data):
    data = np.array(data)
    data_min, data_max = np.min(data), np.max(data)
    return np.ones_like(data) if data_max == data_min else (data - data_min) / (data_max - data_min)

# Objective function for optimization
def objective(x, materials, normalized_properties, constraints, weights):
    composition_penalty = abs(sum(x) - 1) * 1000
    
    properties = ['density', 'strength', 'corrosion_resistance', 'cost', 'sustainability_score']
    property_penalties = sum(
        weights[prop] * (sum(x[i] * normalized_properties[prop][i] for i in range(len(materials))) - normalize([constraints[f'target_{prop}']])[0])**2
        for prop in properties
    )
    
    return composition_penalty + property_penalties

# Optimization of material composition
def optimize_composition(materials, constraints, weights):
    property_names = ['density', 'strength', 'corrosion_resistance', 'cost', 'sustainability_score']
    properties = {prop: [materials[mat][prop] for mat in materials] for prop in property_names}
    normalized_properties = {prop: normalize(properties[prop]) for prop in property_names}
    
    bounds = [(0, 1) for _ in range(len(materials))]
    constraints_eq = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
    
    result = minimize(
        objective, 
        [1/len(materials)] * len(materials), 
        args=(materials, normalized_properties, constraints, weights),
        method='SLSQP', 
        bounds=bounds, 
        constraints=[constraints_eq]
    )
    
    return result.x if result.success else None

# API route for material recommendation
@app.route('/recommend_material', methods=['POST'])
def recommend_material():
    try:
        # Get user input from JSON
        user_constraints = request.json.get('constraints')
        weights = request.json.get('weights')

        # Filter materials based on user-defined cost and sustainability constraints
        filtered_materials = {}
        for name, props in all_materials.items():
            cost = props.get('cost')
            sustainability_score = props.get('sustainability_score')
            
            # Skip materials with missing 'cost' or 'sustainability_score'
            if cost is None or sustainability_score is None:
                continue
            
            # Apply user constraints
            if cost <= user_constraints['target_cost'] and sustainability_score >= user_constraints['target_sustainability']:
                filtered_materials[name] = props
        
        if not filtered_materials:
            return jsonify({"message": "No materials meet the initial cost and sustainability constraints."}), 400
        
        # Find the optimal composition
        best_composition = None
        best_score = float('inf')
        best_materials = None
        
        # Try combinations of 2 and 3 materials
        for r in [2, 3]:
            for materials_subset in combinations(filtered_materials, r):
                materials = {name: filtered_materials[name] for name in materials_subset}
                composition = optimize_composition(materials, user_constraints, weights)
                
                if composition is not None:
                    score = objective(
                        composition, 
                        materials, 
                        {prop: normalize([materials[mat][prop] for mat in materials]) for prop in weights.keys()},
                        user_constraints, 
                        weights
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_composition = composition
                        best_materials = materials
        
        if best_composition is not None:
            result = {
                "optimal_composition": {
                    mat: f"{percentage * 100:.2f}%" for mat, percentage in zip(best_materials.keys(), best_composition)
                },
                "final_score": best_score
            }
            return jsonify(result), 200
        else:
            return jsonify({"message": "Optimization failed to find a solution."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_materials()  # Load materials when the app starts
    app.run(debug=True)
