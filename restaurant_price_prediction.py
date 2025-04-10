# Restaurant Price Prediction Project

# === 1. Import Libraries ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# === 2. Load Sample Dataset ===
df = pd.read_csv("zomato.csv", encoding='latin-1')


# === 3. Data Cleaning & Preprocessing ===
df = df.drop_duplicates()
df = df[df['rate'] != 'NEW']
df = df[df['rate'].notnull()]
df['rate'] = df['rate'].apply(lambda x: x.split('/')[0].strip())
df['rate'] = df['rate'].astype(float)
df = df[df['approx_cost(for two people)'].notnull()]
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).apply(lambda x: x.replace(',', ''))
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)

# Selecting relevant columns
data = df[['rate', 'location', 'rest_type', 'cuisines', 'online_order', 'book_table', 'approx_cost(for two people)']]
data = data.dropna()

# Encode categorical columns
le = LabelEncoder()
data['location'] = le.fit_transform(data['location'])
data['rest_type'] = le.fit_transform(data['rest_type'])
data['cuisines'] = le.fit_transform(data['cuisines'])
data['online_order'] = le.fit_transform(data['online_order'])
data['book_table'] = le.fit_transform(data['book_table'])

# Features and target
X = data.drop('approx_cost(for two people)', axis=1)
y = data['approx_cost(for two people)']

# === 4. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Model Training ===
model = RandomForestRegressor()
model.fit(X_train, y_train)

# === 6. Evaluation ===
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# === 7. Streamlit Web App ===
def run_app():
    st.title("Restaurant Price Prediction")
    
    rate = st.slider("Restaurant Rating (0.0 - 5.0)", 0.0, 5.0, 3.5, 0.1)
    location = st.text_input("Location")
    rest_type = st.text_input("Restaurant Type")
    cuisines = st.text_input("Cuisine Type")
    online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
    book_table = st.selectbox("Book Table Available?", ["Yes", "No"])

    if st.button("Predict Price"):
        location_enc = le.transform([location])[0] if location in le.classes_ else 0
        rest_type_enc = le.transform([rest_type])[0] if rest_type in le.classes_ else 0
        cuisines_enc = le.transform([cuisines])[0] if cuisines in le.classes_ else 0
        online_order_enc = 1 if online_order == "Yes" else 0
        book_table_enc = 1 if book_table == "Yes" else 0

        input_data = [[rate, location_enc, rest_type_enc, cuisines_enc, online_order_enc, book_table_enc]]
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted Price for Two: ₹{predicted_price:.2f}")

# Uncomment this to run Streamlit locally
# if __name__ == '__main__':
#     run_app()
