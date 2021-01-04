
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from dataManipulator import loadtraindata, loadtestdata, splitdata
import numpy as np
import time


if __name__ == '__main__':

    ###### Hyper Parameters #######
    KNeighbour= [2,3,4,5,6,7,8]
    #######

    X,y = loadtraindata()
    X_test, y_test = loadtestdata()
    print("Finish loading {} data points")
    # for split in splitdata(X, y, 4):
    Xtrain, ytrain,XVal, yval = splitdata(X,y,8)
    print("Running K Nearest Neighbour on {} training samples and {} Validation samples".format(len(ytrain), len(yval)))
    train_time_array = []
    predict_time_array = []
    accuracy = []
    best_K = 2
    best_acc = -1
    for k in KNeighbour:
        print("Running on {} Nearest Neighbour".format(k))
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Xtrain, ytrain)
        train_time = time.time() - start
        train_time_array.append(train_time)
        start = time.time()
        predict = knn.predict(XVal)
        predict_time = time.time() - start
        print('Validation Data train time : {}\t predict time : {}'.format(train_time, predict_time))
        incorrect = np.count_nonzero(np.subtract(predict, yval))
        result= 1 - incorrect / len(predict)
        print('Validation Result')
        print(result)
        start = time.time()
        ypredict = knn.predict(X_test)
        predict_time = time.time() - start
        predict_time_array.append(predict_time)
        print('Test Data train time : {}\t predict time : {}'.format(train_time, predict_time))
        incorrect = np.count_nonzero(np.subtract(ypredict, y_test))
        acc= 1- incorrect / len(ypredict)
        print('Test Result')
        print(acc)
        accuracy.append(acc)
        if best_acc < acc:
            best_k = k
            best_acc = acc
    print("Train Times = ", train_time_array)
    print("Predict Time = ", predict_time_array)
    print("Accuracy", accuracy)
    print("Best Accuracy", best_acc)
    print("Best K", best_k)

    plt.plot(KNeighbour, accuracy)
    plt.xlabel('Neighbours')
    plt.ylabel('Test Accuracy')
    plt.savefig('KNN.png')
