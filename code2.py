import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
import matplotlib.pyplot as plt

# Load and preprocess data function
def load_and_preprocess_data(file_path):
    # Load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Convert JSON to DataFrame
    df = pd.DataFrame.from_dict(data['Material'], orient='index')
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Normalize numerical columns
    scaler = MinMaxScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    return df_encoded, scaler

# Calculate sustainability score function
def calculate_sustainability_score(df):
    sustainability_factors = [
        'Recyclability', 'Carbon Footprint', 'Sustainable Sourcing',
        'Resource Depletion', 'Biodegradability'
    ]
    
    # Ensure all factors are present in the DataFrame
    present_factors = [f for f in sustainability_factors if f in df.columns]
    
    # Calculate the score
    sustainability_score = df[present_factors].mean(axis=1)
    
    # Normalize the score
    sustainability_score = (sustainability_score - sustainability_score.min()) / (sustainability_score.max() - sustainability_score.min())
    
    return sustainability_score

# Create and train model function
def create_and_train_model(X_train, y_train, X_test, y_test):
    # Convert data to float32
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=100, batch_size=32, verbose=1)
    
    return model, history

# Material recommendation based on sustainability score
def recommend_sustainable_material(df, car_part):
    # Filter materials based on recommended car parts
    suitable_materials = df[df.index.str.contains(car_part, case=False)]
    
    if suitable_materials.empty:
        return "No suitable materials found for the specified car part."
    
    # Get the material with the highest sustainability score
    best_material = suitable_materials.loc[suitable_materials['Sustainability_Score'].idxmax()]
    
    return best_material.name

# Material recommendation based on custom properties
def recommend_material_with_properties(model, scaler, df, car_part, properties):
    # Filter materials based on recommended car parts
    suitable_materials = df[df.index.str.contains(car_part, case=False)]
    
    if suitable_materials.empty:
        return "No suitable materials found for the specified car part."
    
    # Prepare user input for prediction
    user_input = np.zeros(len(df.columns))
    for prop, value in properties.items():
        if prop in df.columns:
            user_input[df.columns.get_loc(prop)] = value
    
    # Scale user input
    user_input_scaled = scaler.transform([user_input])
    
    # Predict sustainability scores
    predicted_scores = model.predict(suitable_materials.drop('Sustainability_Score', axis=1))
    
    # Combine predicted scores with actual materials
    materials_with_scores = pd.DataFrame({
        'Material': suitable_materials.index,
        'Predicted_Score': predicted_scores.flatten()
    })
    
    # Sort by predicted score and get the top recommendation
    best_material = materials_with_scores.sort_values('Predicted_Score', ascending=False).iloc[0]
    
    return best_material['Material']

# Get user input for car part and properties
def get_user_input(df):
    car_part = input("Enter the car part you're looking for a material for: ")
    
    print("\nAvailable properties:")
    for col in df.columns:
        if col != 'Sustainability_Score':
            print(f"- {col}")
    
    properties = {}
    while True:
        prop = input("\nEnter a property name (or press Enter to finish): ")
        if prop == "":
            break
        if prop not in df.columns:
            print("Invalid property name. Please try again.")
            continue
        value = float(input(f"Enter the desired value for {prop}: "))
        properties[prop] = value
    
    return car_part, properties

# Main material recommendation system function
def material_recommendation_system(model, scaler, df):
    car_part, properties = get_user_input(df)
    
    print("\nInitial recommendation based on sustainability:")
    sustainable_recommendation = recommend_sustainable_material(df, car_part)
    print(f"Most sustainable material for {car_part}: {sustainable_recommendation}")
    
    if properties:
        print("\nRefined recommendation based on specified properties:")
        refined_recommendation = recommend_material_with_properties(model, scaler, df, car_part, properties)
        print(f"Recommended material for {car_part} with specified properties: {refined_recommendation}")
    
    print("\nDetailed information:")
    if properties:
        material_info = df.loc[refined_recommendation]
    else:
        material_info = df.loc[sustainable_recommendation]
    
    for prop, value in material_info.items():
        print(f"{prop}: {value}")

# Plot training history
def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    df, scaler = load_and_preprocess_data('sample dataset.json')
    
    # Calculate sustainability score
    df['Sustainability_Score'] = calculate_sustainability_score(df)
    
    # Prepare data for the model
    X = df.drop(['Sustainability_Score'], axis=1)
    y = df['Sustainability_Score']
    
    # Check if we have enough data to split
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the model
        model, history = create_and_train_model(X_train, y_train, X_test, y_test)
        
        # Plot training history
        plot_training_history(history)
        
        # Run the material recommendation system
        material_recommendation_system(model, scaler, df)
    else:
        print("Not enough data to train the model. Please provide more samples in the dataset.")
