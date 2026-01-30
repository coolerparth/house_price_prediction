import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # Fixed: changed plt to px to match usage below
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

def train_model():
    df = generate_house_data(n_samples=100)
    # Fixed: Reshaping X to 2D array as required by Scikit-Learn
    X = df[['size']] 
    Y = df['price']
    # Fixed: test_size must be between 0.0 and 1.0 (2.0 was impossible)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model, df # Returning df so we can plot it later

def main():
    st.title("Simple Linear Regression House price prediction")
    st.write("Put in your house size to know its price")

    # Fixed: unpack both the model and the dataframe
    model, df = train_model()

    size = st.number_input('House size', min_value=500, max_value=2000, value=1500)

    if st.button('Predict price'):
        predicted_price = model.predict([[size]])
        # Fixed: changed curly brace {0} to square bracket [0]
        st.success(f'Estimated price: ${predicted_price[0]:,.2f}')

        # Fixed: px.scatter used correctly (px was imported as plt before)
        fig = px.scatter(df, x='size', y='price', title="Size vs House Price")
        
        fig.add_scatter(x=[size], y=[predicted_price[0]], 
                        mode='markers',
                        # Fixed: dict uses () and size key was misspelled
                        marker=dict(size=15, color='red'),
                        name='Prediction')
        
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()