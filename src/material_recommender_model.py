from dataset_preprocessing import load_material_data, preprocess_data
from utils import filter_materials

def recommend_material(user_input, dataset_file="../data/materials.json"):
    df = load_material_data(dataset_file)
    df = preprocess_data(df)
    recommendations = filter_materials(df, user_input)
    
    return recommendations[['Material', 'Similarity', 'Recommended Parts']].head(5)

if __name__ == "__main__":
    user_input = {
        # Example user input
        "crashworthiness": 0,
        "corrosion_resistance": 130,
        "impact_resistance": 80,
    }
    recommendations = recommend_material(user_input)
    print(recommendations)
