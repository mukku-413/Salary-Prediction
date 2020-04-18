import os

os.chdir(r"C:\Users\MukeshKumar\PycharmProjects\Project1")
import pandas as pd
import pickle

dataset = pd.read_csv("hiring.csv")
dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]


# converting words into integer values
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                 'eleven': 11, 'twelve': 12, 'zero': 0, 0:0}
    return word_dict[word]


X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))
y = dataset.iloc[:, -1]

# training the model on entire data as we have very small number of training examples
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)

# saving the trained model
pickle.dump(model, open('model.pkl', 'wb'))

# loading the model to make predictions
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2, 9, 6]]))
