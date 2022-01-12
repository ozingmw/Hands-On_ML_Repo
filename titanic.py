from file_load import csv

train_data = csv.load("./datasets/titanic/train.csv")
test_data =csv.load("./datasets/titanic/test.csv")

print(train_data.head(10))