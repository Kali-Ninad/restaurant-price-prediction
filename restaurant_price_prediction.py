# Restaurant Price Prediction Project (Updated with Your Dataset)

# === 1. Import Libraries ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# === 2. Load Dataset ===
df = pd.read_csv("zomato.csv", encoding='latin-1')
df = df.drop_duplicates()
df = df[df['Aggregate rating'] > 0]
df = df[df['Average Cost for two'].notnull()]

# === 3. Select and Rename Relevant Columns ===
df = df[['Aggregate rating', 'City', 'Cuisines', 'Has Table booking', 'Has Online delivery', 'Average Cost for two']]
df = df.rename(columns={
    'Aggregate rating': 'rate',
    'City': 'location',
    'Has Table booking': 'book_table',
    'Has Online delivery': 'online_order',
    'Average Cost for two': 'cost'
})

# === 4. Encode Categorical Columns ===
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])
df['Cuisines'] = le.fit_transform(df['Cuisines'].astype(str))
df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})
df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})

# === 5. Features and Target ===
X = df[['rate', 'location', 'Cuisines', 'online_order', 'book_table']]
y = df['cost']

# === 6. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 7. Model Training ===
model = RandomForestRegressor()
model.fit(X_train, y_train)

# === 8. Evaluation ===
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# === 9. Streamlit Web App ===
def run_app():
    st.title("Restaurant Price Prediction")

    rate = st.slider("Restaurant Rating (0.0 - 5.0)", 0.0, 5.0, 3.5, 0.1)
    location = st.text_input("City")
    cuisines = st.text_input("Cuisine Type")
    online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
    book_table = st.selectbox("Book Table Available?", ["Yes", "No"])

    if st.button("Predict Price"):
        location_enc = le.transform([location])[0] if location in le.classes_ else 0
        cuisines_enc = le.transform([cuisines])[0] if cuisines in le.classes_ else 0
        online_order_enc = 1 if online_order == "Yes" else 0
        book_table_enc = 1 if book_table == "Yes" else 0

        input_data = [[rate, location_enc, cuisines_enc, online_order_enc, book_table_enc]]
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted Price for Two: ₹{predicted_price:.2f}")

# Uncomment this to run Streamlit locally
if __name__ == '__main__':
     run_app()
