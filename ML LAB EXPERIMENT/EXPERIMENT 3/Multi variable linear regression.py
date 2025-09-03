import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

colnames=['areas','rooms','prices']
dataset = pd.read_csv("https://raw.githubusercontent.com/nishithkotak/machine-learning/refs/heads/master/ex1data2.txt",names=colnames)

dataset.describe()

areas=dataset.iloc[0:dataset.shape[0],0:1]
romms=dataset.iloc[0:dataset.shape[0],1:2]
prices=dataset.iloc[0:dataset.shape[0],2:3]

dataset.shape

from posixpath import splitdrive
#function normalization
def feature_normalization(x):
  mean=np.mean(x,axis=0)
  std=np.std(x,axis=0)
  x_normalized=(x-mean)/std
  return x_normalized,mean,std

data_norm = dataset.values
m = data_norm.shape[0]
#taking features vectors
x2 = data_norm[:, 0:2].reshape(m, 2)
x2_norm, mean, std = feature_normalization(x2)

y2 = data_norm[:, 2:3].reshape(m, 1)

x2_norm

theta_array=np.zeros((3,1))

def Hypothesis(theta_array , x1 , x2) :
  return theta_array[0] + theta_array[1]*x1 + theta_array[2]*x2

def Cost_Function(theta_array,x1,x2,y,m):
  total_cost = 0
  for i in range(m):
    total_cost += (Hypothesis(theta_array,x1[i] , x2[i]) - y[i])**2
    return total_cost/(2*m)

def Gradient_Descent(theta_array , x1, x2, y , m ,alpha) :
  summation_0 = 0
  summation_1 = 0
  summation_2 = 0
  for i in range(m):
    summation_0 += (Hypothesis(theta_array,x1[i] , x2[i]) - y[i])
    summation_1 += ((Hypothesis(theta_array,x1[i] , x2[i]) - y[i])*x1[i])
    summation_2 += ((Hypothesis(theta_array,x1[i] , x2[i]) - y[i])*x2[i])
    new_theta0 = theta_array[0] - (alpha/m)*summation_0
    new_theta1 = theta_array[1] - (alpha/m)*summation_1
    new_theta2 = theta_array[2] - (alpha/m)*summation_2
    new_theta = [new_theta0 , new_theta1 , new_theta2]
    return new_theta

def Training(x1, x2, y, alpha, iters):
  theta_0 = 0
  theta_1 = 0
  theta_2 = 0
  theta_array = [theta_0, theta_1 ,theta_2]
  m = len(x1)
  cost_values = []
  for i in range(iters):
    theta_array = Gradient_Descent(theta_array, x1 ,x2, y, m, alpha)
    loss = Cost_Function(theta_array, x1 ,x2, y, m)
    cost_values.append(loss)
    y_new = theta_array[0] + theta_array[1]*x1 + theta_array[2]*x2
    return theta_array , cost_values

def Training(x1, x2, y, alpha, iters):
  theta_0 = 0
  theta_1 = 0
  theta_2 = 0
  theta_array = [theta_0, theta_1 ,theta_2]
  m = len(x1)
  cost_values = []
  for i in range(iters):
    theta_array = Gradient_Descent(theta_array, x1 ,x2, y, m, alpha)
    loss = Cost_Function(theta_array, x1 ,x2, y, m)
    cost_values.append(loss)
    y_new = theta_array[0] + theta_array[1]*x1 + theta_array[2]*x2
    return theta_array , cost_values

alpha = 0.01
iters = 500

area_norm = x2_norm[:, 0]
room_norm = x2_norm[:, 1]
price_norm = y2
theta_array, cost_per_itr = Training(area_norm, room_norm, price_norm, alpha, iters)
predicted_price = theta_array[0] + theta_array[1]*area_norm + theta_array[2]*room_norm

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(area_norm, room_norm, price_norm, alpha=0.3, c='#FF0000', label="Actual")
ax.plot(area_norm, room_norm, predicted_price, c="#0000FF", label="Predicted")
ax.set_xlabel("Area")
ax.set_ylabel("Rooms")
ax.set_zlabel("Prices")
ax.set_title("Best Fit Line")
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.subplot(3, 1, 1)
sns.scatterplot(x='areas', y='rooms', data=dataset,palette='prices')
plt.title('Area vs Prices')
plt.xlabel('Area (sq ft)')
plt.ylabel('Prices ($)')
plt.subplot(3, 1, 2)
sns.scatterplot(x='rooms', y='prices', data=dataset, palette='viridis')
plt.title('Rooms vs Prices')
plt.xlabel('Number of Rooms')
plt.ylabel('Prices ($)')
plt.subplot(3, 1, 3)
sns.scatterplot(x='rooms', y='areas', data=dataset, palette='viridis')
plt.title('Rooms vs Area')
plt.xlabel('Number of Rooms')
plt.ylabel('Area (sq ft)')
plt.tight_layout()
plt.show()