import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(steps):
        dj_dw = np.zeros((n,))
        dj_db = 0

        for i in range(m):

            z = np.dot(w, X[i]) + b
            f_wb_i = _sigmoid(z)

            err_i = f_wb_i - y[i]

            for j in range(n):
                dj_dw[j] += err_i * X[i, j]
            dj_db += err_i

        dj_dw = dj_dw / m
        dj_db = dj_db / m


        w = w - (lr * dj_dw)
        b = b - (lr * dj_db)

    return (w, b)
    

    
        