import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Read CSV file with space delimiter
df = pd.read_csv('Earthquake_Data.csv', delimiter=r'\s+')

new_column_names = ["Date(YYYY/MM/DD)",  "Time(UTC)", "Latitude(deg)", "Longitude(deg)", "Depth(km)", "Magnitude(ergs)", 
                    "Magnitude_type", "No_of_Stations", "Gap", "Close", "RMS", "SRC", "EventID"]

df.columns = new_column_names
ts = pd.to_datetime(df["Date(YYYY/MM/DD)"] + " " + df["Time(UTC)"])
df = df.drop(["Date(YYYY/MM/DD)", "Time(UTC)"], axis=1)
df.index = ts

# Select relevant columns
X = df[['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'No_of_Stations']]
y = df['Magnitude(ergs)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize a random forest regressor with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the regressor to the training data
rf.fit(X_train, y_train)

# Predict on the testing set
y_pred = rf.predict(X_test)

# Compute R^2 and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Streamlit App
st.set_page_config(page_title='Earthquake Prediction App', layout='wide')

st.title('Earthquake Prediction App')

# Input form
latitude = st.number_input('Enter Latitude:', min_value=-90.0, max_value=90.0, value=0.0)
longitude = st.number_input('Enter Longitude:', min_value=-180.0, max_value=180.0, value=0.0)
depth = st.number_input('Enter Depth (km):', min_value=0.0, value=0.0)
nos = st.number_input('Enter Number of Stations:', min_value=0, value=0)

# Predict on user input
if st.button('Predict'):
    input_data = [[latitude, longitude, depth, nos]]
    predicted_growth = rf.predict(input_data)[0]
    
    st.header('Prediction Result:')
    st.write(f'Predicted Magnitude: {predicted_growth:.2f}')
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R^2 Score: {r2:.2f}')
    st.write(f'Mean Absolute Error: {mae:.2f}')# Add this changes
