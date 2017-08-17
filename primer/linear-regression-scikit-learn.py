import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
df = pd.read_fwf('../data/primer/brain_body.txt')
# print(df.head())
xValues = df[['Brain']]
yValues = df[['Body']]

# train model on data
bodyReg = linear_model.LinearRegression()
bodyReg.fit(xValues, yValues)

# visualize the results
plt.scatter(xValues, yValues)
plt.plot(xValues, bodyReg.predict(xValues))
plt.show()

# play with predictions
print(bodyReg.predict([127]))