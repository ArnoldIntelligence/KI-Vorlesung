
import numpy as np
import matplotlib.pyplot as plt

def f_id(z):
    return z

def f_relu(z):
    # result = np.zeros_like(z)
    # result(z >= 0, z)
    # return result
    return np.maximum(0, z)

def f_drelu(z):
    return np.array(z>=0).astype(int)

def f_sigmoid(z):
    return 1/(1+np.exp(-z))

def f_dsigmoid(z):
    t = 1/(1+np.exp(-z))
    return t*(1-t)

def add_bias_column(A):
    """Hängt eine 1-Spalte links an: (m,n) -> (m,n+1), Bias = x0 = 1."""
    return np.hstack([np.ones((A.shape[0], 1)), A])

# forward-pass
def predict(X,WIH,WHO):

    X1 = add_bias_column(X)     # Bias anhängen
    z1 = X1 @ WIH               # Gewichtete Summe, @ ist Kurzschreibweise für np.matmul oder np.dot
    a1 = f_relu(z1)          # Aktivierungsfunktion
    # a1 = f_sigmoid(z1)          # Aktivierungsfunktion
    h = add_bias_column(a1)     # Bias anhängen
    z2 = h @ WHO                # Gewichtete Summe
    a2 = f_sigmoid(z2)             # Outputfunktion

    cache = (X1, z1, a1, h, z2, a2)    # Werte zwischenspeichern

    return a2, cache

def fit(X, y, WIH, WHO, n_iter, alpha=1.0):
    y = y.reshape(-1, 1)   # sicherstellen, dass y ein Spaltenvektor ist
    losses = []

    for i in range(n_iter):
        # 1) Forward Pass
        a2, cache = predict(X, WIH, WHO)
        X1, z1, a1, h, z2, a2 = cache

        # 2) Loss berechnen
        loss = 0.5 * np.mean((y - a2) ** 2)
        losses.append(loss)

        # 3) Output-Delta
        delta_O = (a2 - y) * f_dsigmoid(z2)

        # 4) Gradient für WHO
        grad_WHO = h.T @ delta_O

        # 5) Hidden-Delta
        WHO_no_bias = WHO[1:, :]          # Bias-Gewichte entfernen
        delta_H = (delta_O @ WHO_no_bias.T) * f_drelu(z1)
        # delta_H = (delta_O @ WHO_no_bias.T) * f_dsigmoid(z1)

        # 6) Gradient für WIH
        grad_WIH = X1.T @ delta_H

        # 7) Update
        N = X.shape[0]
        WIH = WIH - alpha * grad_WIH / N
        WHO = WHO - alpha * grad_WHO / N

        # 8) Ausgabe
        if i % 1000 == 0:
            print(i, loss)

    return WIH, WHO, losses

def plot(X, y, WIH, WHO):

    y = y.ravel()

    x_min, x_max = X[:,0].min()-0.2, X[:,0].max()+0.2
    y_min, y_max = X[:,1].min()-0.2, X[:,1].max()+0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    zz, _ = predict(grid, WIH, WHO)
    zz = zz.reshape(xx.shape)

    cont = plt.contourf(xx, yy, zz, levels=30, alpha=0.7, cmap="RdGy")

    colors = {1: "black", 0: "red"}

    for cls in np.unique(y):
        mask = (y == cls)
        plt.scatter(X[mask,0],X[mask,1],color=colors[int(cls)],edgecolor="white",s=100,label=f"class {int(cls)}"
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Entscheidungsfläche des Netzes")
    plt.legend()
    plt.colorbar(cont)

 
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([0, 1, 1, 0], dtype=float).reshape(-1, 1)

np.random.seed(0)
WIH = np.random.randn(3,4) * 1.0
WHO = np.random.randn(5,1) * 1.0

WIH, WHO, losses = fit(X, y, WIH, WHO, n_iter=1000, alpha=1.0)

pred, _ = predict(X, WIH, WHO)
print(pred)
print(losses[-1])

print(WIH, WHO)

plot(X, y, WIH, WHO)

plt.figure(figsize=(6,5))
plt.plot(losses)
plt.xlabel('no of iteration')
plt.ylabel('loss')
plt.show()
