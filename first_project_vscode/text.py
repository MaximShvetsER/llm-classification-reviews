import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("gender_submission.csv", sep = ",")
PassendgerID = df["PassengerId"]
PassendgerID_spisok = []
for i in PassendgerID:
    PassendgerID_spisok.append(i)
ind_survived = []
for i in df["Survived"]:
    ind_survived.append(i)

model = LogisticRegression()
x = []
for i in range(10**10):
    x.append([[PassendgerID[i], ind_survived[i]]])
print(x)



