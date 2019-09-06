import pandas

#load predefined LinearRegression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import visualization lib
import matplotlib.pyplot as plt

#load data
df = pandas.read_csv("./HousePrice.csv")
df = df[["Price (Older)", "Price (New)"]] #the table has much column, you just need a old and new price tu use

X = df[["Price (Older)"]] #X is old price
Y = df[["Price (New)"]] #Y is new price

x_train, x_test, y_train, y_test = train_test_split(X, Y) #this is split a value into 70:30 (training:testing)

lr = LinearRegression().fit(x_train, y_train) #lr = linear regression

# formula that we use is ((W * x) + b)
print("Coef (W1) : ", lr.coef_)
print("Intercept : ", lr.intercept_)

W1 = lr.coef_ # gradient/slope/weight
print(W1)
b = lr.intercept_ #b is bias

plt.scatter(X, Y)
plt.plot(X, W1*X+b, "r-")
plt.show()



