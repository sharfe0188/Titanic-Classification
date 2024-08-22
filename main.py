import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("Titanic Survival Prediction")

# Load Data
@st.cache
def load_data():
    data = pd.read_csv("C:\\Users\\mdsha\\Desktop\\Titanic Classification\\titanic.csv")
    return data

data = load_data()

# Display Dataset
st.subheader("Titanic Dataset")
st.write(data.head())

# Preprocessing
st.subheader("Preprocessing")
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna('S', inplace=True)

# Convert categorical columns to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Feature selection
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar for user input
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of trees in the forest", 1, 100, 10)
max_depth = st.sidebar.slider("Maximum depth of the tree", 1, 20, 5)

# Train the model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display the results
st.subheader("Model Accuracy")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader("Feature Importance")
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
st.write(feature_importance.sort_values(by='Importance', ascending=False))

# Prediction on user input
st.subheader("Predict Survival")
def user_input_features():
    pclass = st.sidebar.selectbox('Pclass', (1, 2, 3))
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    age = st.sidebar.slider('Age', 0, 80, 29)
    sibsp = st.sidebar.slider('SibSp', 0, 8, 0)
    parch = st.sidebar.slider('Parch', 0, 6, 0)
    fare = st.sidebar.slider('Fare', 0, 500, 32)
    embarked = st.sidebar.selectbox('Embarked', ('S', 'C', 'Q'))
    sex = 0 if sex == 'male' else 1
    embarked = 0 if embarked == 'S' else (1 if embarked == 'C' else 2)

    data = {'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

prediction = model.predict(input_df)

st.subheader("Prediction Result")
st.write("Survived" if prediction[0] == 1 else "Not Survived")
