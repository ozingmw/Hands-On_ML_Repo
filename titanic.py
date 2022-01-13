from file_load import csv

import matplotlib.pyplot as plt

train_data = csv.load("./datasets/titanic/train.csv")
test_data =csv.load("./datasets/titanic/test.csv")

print(train_data.head(10))
print(train_data.info())

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

train_data.drop