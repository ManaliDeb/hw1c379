import pandas as pd
import numpy as np

data = pd.read_csv('src/main/resources/train.csv')

def preprocess_data(data):
    # convert sex to 0 and 1
    sex_map = {'male': 0, 'female': 1}
    data['Sex'] = data['Sex'].map(sex_map)

    # replace missing age with nan
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

    # find median age
    median_age = data['Age'].median()
    print(f"Median Age: {median_age}")
    if pd.isna(median_age):
        raise ValueError("median age calculation failed")

    # put median age for missing age
    data['Age'] = data['Age'].fillna(median_age)

    # handle fares
    data['Fare'] = pd.to_numeric(data['Fare'], errors='coerce')
    median_fare = data['Fare'].median()
    print(f"Median Fare: {median_fare}")

    # cap fare value and put nan for missing values
    data['Fare'] = data['Fare'].fillna(median_fare)
    data['Fare'] = data['Fare'].clip(upper=500)

    # normalize the data
    data['Age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
    data['Fare'] = (data['Fare'] - data['Fare'].min()) / (data['Fare'].max() - data['Fare'].min())

    # convert features to float
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    features = data[feature_cols].values
    labels = data['Survived'].values  # survived is target

    return features, labels

# preprocess data
features, labels = preprocess_data(data)

# perceptron model, gradient descent
def train_perceptron(features, labels, epochs=1000, lr=0.01):
    n_samples, n_features = features.shape
    weights = np.zeros(n_features)  # initialize
    bias = 0

    for epoch in range(epochs):
        for i in range(n_samples):
            linear_output = np.dot(features[i], weights) + bias
            y_pred = np.where(linear_output >= 0, 1, 0)

            # update weights
            update = lr * (labels[i] - y_pred)
            weights += update * features[i]
            bias += update

    return weights, bias

# train model
weights, bias = train_perceptron(features, labels)

# predict function
def predict(features, weights, bias):
    linear_output = np.dot(features, weights) + bias
    predictions = np.where(linear_output >= 0, 1, 0)
    return predictions

# evaluate accuracy
predictions = predict(features, weights, bias)
accuracy = np.mean(predictions == labels) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

