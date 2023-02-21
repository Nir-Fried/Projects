import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100

# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1,X2],axis=1)
y = np.concatenate([np.ones((n,1)), -np.ones((n,1))], axis=0).reshape([-1])

def Q1(X,y):
    '''
    Plots the results of the SVM with linear, homogeneous (coef0=0) polynomial of degree 2 and 3 kernels.
    '''
    C = 10
    models = (svm.SVC(kernel='linear', C=C).fit(X,y), # Linear Kernel
                svm.SVC(kernel='poly', degree=2, C=C,coef0=0,gamma='auto').fit(X,y),  # Polynomial Kernel with degree 2
                svm.SVC(kernel='poly', degree=3, C=C,coef0=0,gamma='auto').fit(X,y))  # Polynomial Kernel with degree 3

    titles = ('linear kernel','poly (deg 2) kernel','poly (deg 3) kernel')
    plot_results(models, titles, X, y)

def Q2(X,y):
    '''
    Plots the results of the SVM with linear, non-homogeneous polynomial of degree 2 and 3 kernels.
    '''
    C = 10
    models = (svm.SVC(kernel='linear', C=C).fit(X,y), # Linear Kernel
                svm.SVC(kernel='poly', degree=2, C=C,coef0=1,gamma='auto').fit(X,y),  # Polynomial Kernel with degree 2
                svm.SVC(kernel='poly', degree=3, C=C,coef0=1,gamma='auto').fit(X,y))  # Polynomial Kernel with degree 3

    titles = ('linear kernel','poly (deg 2) kernel','poly (deg 3) kernel')
    plot_results(models, titles, X, y)

def Q3(X,y):
    '''
    Assuming now that 10% of the data points are mislabeled.
    Plots the results of the SVM with RBF and polynomial of degree 2 kernels.
    '''
    C = 10
    for i,val in enumerate(y):
        if val == -1:
            rn = np.random.randint(10)
            if rn == 0: # 10% of the time
                y[i] = 1 
    
    models = (svm.SVC(kernel='rbf',gamma=10,C=C).fit(X,y), 
                svm.SVC(kernel='poly',degree=2,gamma='auto',C=C,coef0=1).fit(X,y)) 
    titles = ('RBF kernel','poly kernel')
    plot_results(models, titles, X, y)

def Q3RBF(X,y):
    '''
    Assuming that 10% of the data points are mislabeled.
    Plots the results of the SVM with RBF kernels with different gamma values.
    '''
    C = 10
    for i,val in enumerate(y):
        if val == -1:
            rn = np.random.randint(10)
            if rn == 0: # 10% of the time
                y[i] = 1 
    
    models = (svm.SVC(kernel='rbf',gamma=1,C=C).fit(X,y), 
                svm.SVC(kernel='rbf',gamma=20,C=C).fit(X,y),
                svm.SVC(kernel='rbf',gamma=50,C=C).fit(X,y)) 
    titles = ('gamma = 1','gamma = 20','gamma = 50')
    plot_results(models, titles, X, y)
Q3RBF(X,y)

