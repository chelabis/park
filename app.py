import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st

def load_data(file_path):
    # Load data from CSV file
    data = pd.read_csv(file_path)
    return data

def prepare_data(data, section_id):
    # Filter the data for the specific section_id
    section_data = data[data['section_id'] == section_id]
    if section_data.empty:
        raise ValueError(f"Section ID {section_id} not found in the data.")
    
    # Extract the prices for the last three years
    years = np.array([1399, 1400, 1401])
    prices = section_data.iloc[0, 1:].values.astype(float)
    
    # Reshape for sklearn
    X = years.reshape(-1, 1)
    y = prices
    
    return X, y

def train_model(X, y):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train the MLP Regressor model
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=50000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def predict_price(model, scaler, year):
    # Predict the price for the given year
    year_scaled = scaler.transform(np.array([[year]]))
    predicted_price = model.predict(year_scaled)
    return predicted_price[0]

def main():
    st.title("Section Price Prediction")

    file_path = st.text_input("File Path", "DB.csv")
    section_id = st.text_input("Section ID", "A122")
    year_to_predict = st.number_input("Year to Predict", min_value=1402, max_value=1405, value=1402, step=1)
    
    if st.button("Run Model"):
        try:
            data = load_data(file_path)
            X, y = prepare_data(data, section_id)
            
            model, scaler = train_model(X, y)
            
            predicted_price = predict_price(model, scaler, year_to_predict)
            st.success(f"The predicted price for section {section_id} in {year_to_predict} is: T{predicted_price:.2f}")
        except Exception as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
