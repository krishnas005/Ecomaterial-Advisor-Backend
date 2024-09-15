import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
from dataset_preprocessing import load_material_data, preprocess_data

def train_model(df, save_model=True):
    features = df.drop(columns=['Material', 'recommended_parts']) 
    model = NearestNeighbors(n_neighbors=5)
    model.fit(features)
    
    if save_model:
        joblib.dump(model, '../models/recommendation_model.pkl')
    return model

def load_model(model_path='../models/recommendation_model.pkl'):
    return joblib.load(model_path)

if __name__ == "__main__":
    # Load and preprocess data
    df = load_material_data("../data/materials.json")
    df = preprocess_data(df)
    
    # Train model
    model = train_model(df)
    print("Model trained and saved.")
