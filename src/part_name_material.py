import json
import os



def load_materials(json_file):
    """
    Load materials data from a JSON file.

    Parameters:
    - json_file (str): Path to the JSON file containing materials data.

    Returns:
    - dict: Dictionary containing materials data.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"The file {json_file} does not exist.")

    with open(json_file, 'r') as file:
        data = json.load(file)
    
    if 'materials' not in data:
        raise KeyError("JSON data does not contain 'materials' key.")
    
    return data['materials']

json_file = '../data/materials.json'

materials = load_materials(json_file)

def recommend_materials(part_name, top_n=3):
    """
    Recommend top N materials based on sustainability rating for a given part.

    Parameters:
    - materials (dict): Dictionary of materials data.
    - part_name (str): The part name for which to recommend materials.
    - top_n (int): Number of top materials to recommend.

    Returns:
    - list of tuples: Each tuple contains (Material Name, Sustainability Rating).
    """
    suitable_materials = []

    for material, attributes in materials.items():
        recommended_parts = attributes.get('recommended_parts', [])
        if part_name in recommended_parts:
            sustainability_rating = attributes.get('sustainability_rating', 0)
            suitable_materials.append((material, sustainability_rating))
    
    if not suitable_materials:
        return []

    # Sort materials by sustainability_rating in descending order
    suitable_materials.sort(key=lambda x: x[1], reverse=True)

    # Return top N materials
    return suitable_materials[:top_n]

def main():
    """
    Main function to execute the material recommendation.
    """
    

    try:
        materials = load_materials(json_file)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading materials data: {e}")
        return

    while True:
        part_name = input("\nEnter the part name for material recommendation (or type 'exit' to quit): ").strip()
        
        if part_name.lower() == 'exit':
            print("Exiting the recommendation system. Goodbye!")
            break

        recommendations = recommend_materials(materials, part_name)

        if not recommendations:
            print(f"No materials found that are recommended for the part '{part_name}'. Please try another part.")
        else:
            print(f"\nTop {len(recommendations)} material(s) recommended for '{part_name}':")
            for idx, (material, rating) in enumerate(recommendations, start=1):
                print(f"{idx}. {material} (Sustainability Rating: {rating})")

if __name__ == "__main__":
    main()
