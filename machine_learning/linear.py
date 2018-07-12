from sklearn import linear_model
import matplotlib.pyplot as plt
features = [[140], [130], [150], [170]]
lables = [5664, 4466, 6200, 8900]
plt.scatter(features, lables, color='blue')
plt.xlabel('hours')
plt.ylabel('price')
clf = linear_model.LinearRegression()
clf = clf.fit(features, lables)
result = clf.predict([[127], [193]])
print(result)
plt.plot([[127], [193]], result, color='black')
plt.show()
