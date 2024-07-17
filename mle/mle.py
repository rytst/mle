import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.join(os.path.dirname(__file__), '..', 'height_weight.txt')

data = np.loadtxt(path)

mu = np.mean(data, axis=0)
cov = np.cov(data, rowvar=False)


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi)**D * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / (-2.0))
    return y


xs = np.arange(150, 200, 0.1)
ys = np.arange(40, 80, 0.1)

X, Y = np.meshgrid(xs, ys)


Z = np.zeros_like(X)


for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)




# plot
fig = plt.figure(figsize=(15, 15))

ax1 = fig.add_subplot(2, 1, 1, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis')


height = data[:, 0]
weight = data[:, 1]

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.scatter(height, weight)
ax2.contour(X, Y, Z)


plt.savefig('mle.png')
