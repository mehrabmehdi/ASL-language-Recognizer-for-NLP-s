
import numpy as np

def loadtraindata(numrows=-1):
    dataset = np.genfromtxt('sign_mnist_train.csv', delimiter=',',skip_header=1)
    if numrows == -1:
        X = dataset[:,1:]
        y = dataset[:,0]
    else:
        randindices = np.random.choice(dataset.shape[0], numrows, replace=False)
        X = dataset[randindices,1:]
        y = dataset[randindices,0]

    np.random.seed(314)
    np.random.shuffle(X)
    np.random.seed(314)
    np.random.shuffle(y)
    return X,y

def loadtestdata(numrows=-1):
    dataset = np.genfromtxt('sign_mnist_test.csv', delimiter=',',skip_header=1)
    if numrows == -1:
        X = dataset[:,1:]
        y = dataset[:,0]
    else:
        randindices = np.random.choice(dataset.shape[0], numrows, replace=False)
        X = dataset[randindices,1:]
        y = dataset[randindices,0]

        np.random.seed(314)
        np.random.shuffle(X)
        np.random.seed(314)
        np.random.shuffle(y)
    return X,y


def splitdata(X,y,k):
    N = len(y)

    idx_count = {}
    for i in range(N):
        try: idx_count[y[i]].append(i)
        except: idx_count[y[i]] = [i]

    i = 0
    train_index = []
    val_index = []
    for _class in idx_count:
        chunk = len(idx_count[_class]) // k
        train_index += idx_count[_class][0: i * chunk] + \
                      idx_count[_class][(i + 1) * chunk: -1]
        val_index += idx_count[_class][i * chunk: (i + 1) * chunk]
    np.random.shuffle(train_index)
    np.random.shuffle(val_index)
    

    return X[train_index], y[train_index],X[val_index], y[val_index]
