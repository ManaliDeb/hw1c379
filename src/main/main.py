import numpy as np

# load dataset
data = np.genfromtxt('src/main/resources/train.csv', delimiter=',', skip_header=1, dtype=None,
                     names=True, encoding='utf-8')

# preview data (first few rows)
print(data[:5])

def preprocess_data(data):
    # convert sex to 0s and 1s
    sex_col = 4
    for i in range(len(data)):
        if data[i][sex_col] == 'male':
            data[i] = list(data[i])
            data[i][sex_col] = 0
        elif data[i][sex_col] == 'female':
            data[i] = list(data[i])
            data[i][sex_col] = 1

    # get rid of null values
    age_col = 5
    ages = [row[age_col] for row in data if row[age_col] != '']
    ages = [float(age) for age in ages]
    median_age = np.median(ages)

    for i in range(len(data)):
        if data[i][age_col] == '':
            data[i][age_col] = median_age

    return data

def predict_survival(passenger):
    pclass = passenger[2]
    sex = passenger[4]

    # if woman in class 1 or 2, more likely to survive
    if sex == 1 and pclass in [1, 2]:
        return 1 # survived
    # if man and in class 3
    elif sex == 0 and pclass == 3:
        return 0 # did not survive
    # default did not survive
    else:
        return 0

    # incorporate age, parch, fare, and sibsp.

for i in range(5):
    prediction = predict_survival(data[i])
    print(f"Passenger {i+1}: Predicted survival: {prediction}")

# eval accuracy
correct = 0
total = len(data)

for passenger in data:
    actual = int(passenger[1])  # survival column
    predicted = predict_survival(passenger)
    if actual == predicted:
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
