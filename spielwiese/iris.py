import numpy as np
import matplotlib.pyplot as plt
from perceptron_learn import perceptron_learn

sepal_length = []
sepal_width = []
petal_length = []
petal_width = []
colors = []
colors_values = []
iris_colors = {'Iris-setosa' : 'red', 'Iris-versicolor' : 'green' , 'Iris-virginica' : 'blue'}
iris_colors_values = {'Iris-setosa' : '0', 'Iris-versicolor' : '1' , 'Iris-virginica' : '1'}  ## Anmerkung: rot und blau/gruen lassen sich linear trennen, blau und gruen nicht

with open('../notebooks/data/iris.csv', 'r') as fobj:
    for line in fobj:
        words = line.rstrip().split(',')
        if len(words) != 5:
            continue
        #print(words)
        sepal_length.append(float(words[0]))
        sepal_width.append(float(words[1]))
        petal_length.append(float(words[2]))
        petal_width.append(float(words[3]))
        colors.append(iris_colors[words[4]])
        colors_values.append(int(iris_colors_values[words[4]]))

# plt.style.use('seaborn-v0_8')
# plt.figure(figsize=(12,5))
# ax1 = plt.subplot(1,2,1)
# ax1.set_xlabel('Sepal length')
# ax1.set_ylabel('Sepal Width')
# ax1.set_title('Iris Sepal Length vs. Width')
# ax1.scatter(np.array(sepal_length), np.array(sepal_width), color = colors)
# ax2 = plt.subplot(1,2,2)
# ax2.set_xlabel('Petal length')
# ax2.set_ylabel('Petal Width')
# ax2.set_title('Iris Petal Length vs. Width')
# ax2.scatter(np.array(petal_length), np.array(petal_width), color = colors)
# plt.show()

X = np.transpose(np.array([np.ones(len(sepal_length)), petal_length, petal_width]))
y = colors_values
# weights_initial = np.ones(3)/5
weights_initial = np.array([0.2, 0.5, 0.4])

np.random.seed(42)

# print(np.shape(X))
# print(y)

iterations = 100000

errors, weights = perceptron_learn(X, y, weights_initial, iterations)

print("Gewichte nach Training:", weights[-1])
plt.plot(errors)
# plt.show()

x_vals = np.linspace(min(petal_length), max(petal_length), 100)
w0, w1, w2 = weights[-1]
y_vals = -(w0 + w1 * x_vals) / w2

plt.style.use('seaborn-v0_8')
plt.figure(figsize=(6,5))
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Sepal Length vs. Width')
plt.scatter(np.array(petal_length), np.array(petal_width), color = colors, label = "Iris Messungen")
plt.plot(x_vals, y_vals, color='black', linewidth=2, label="Trennlinie")
plt.show()
plt.legend()
