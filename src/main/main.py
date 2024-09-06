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

