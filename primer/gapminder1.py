import pandas as pd
from sklearn import linear_model

bmo_life_data = pd.read_csv("../data/primer/bmi_and_life_expectancy")
# print(bmo_life_data.head())


# fit the linear regression model
# countryValues = bmo_life_data[["Country"]]
lifeExpectancyValues = bmo_life_data[["Life expectancy"]]
bmiValues = bmo_life_data[["BMI"]]
bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(bmiValues, lifeExpectancyValues, )

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)

print(laos_life_exp)


