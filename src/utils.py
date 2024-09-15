
def calculate_similarity(user_input, material_properties, weight_map=None):
    if weight_map is None:
        weight_map = {}

    similarity_score = 0
    for prop, value in user_input.items():
        if prop in material_properties:
            try:
                # Convert value to float for numerical operations
                value = float(value)
                material_value = float(material_properties[prop])
                weight = weight_map.get(prop, 1)
                similarity_score += (1 - abs(value - material_value) / max(1, material_value)) * weight
            except (ValueError, TypeError):
                # If conversion fails or material_value is not a number, skip this property
                continue
    return similarity_score



def filter_materials(df, user_input, weight_map=None):
    df['Similarity'] = df.apply(lambda row: calculate_similarity(user_input, row.to_dict(), weight_map), axis=1)
    return df.sort_values(by='Similarity', ascending=False)
