
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from dataManipulator import loadtraindata, loadtestdata, splitdata
import numpy as np
import time


if __name__ == '__main__':

    ###### Hyper Parameters #######
    c_param = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 , 0.9, 1]


    #######

    X,y = loadtraindata()
    X_test, y_test = loadtestdata()
    print("Finish loading {} data points")
    # for split in splitdata(X, y, 4):
    Xtrain, ytrain,XVal, yval = splitdata(X,y,8)
    print("Running Logistic Regression on {} training samples and {} Validation samples".format(len(ytrain), len(yval)))
    train_time_array = []
    predict_time_array = []
    accuracy = []
    best_c = 1
    best_acc = -1
    for c in c_param:
        print("Running on {} Regularization Strength".format(c))
        start = time.time()
        LR=LogisticRegression(C = c ,random_state=0, solver='sag', multi_class='multinomial', max_iter=100)
        LR.fit(Xtrain, ytrain)
        train_time = time.time() - start
        train_time_array.append(train_time)
        start = time.time()
        predict = LR.predict(XVal)
        predict_time = time.time() - start
        print('Validation Data train time : {}\t predict time : {}'.format(train_time, predict_time))
        incorrect = np.count_nonzero(np.subtract(predict, yval))
        result= 1 - incorrect / len(predict)
        print('Validation Result')
        print(result)
        start = time.time()
        ypredict = LR.predict(X_test)
        predict_time = time.time() - start
        predict_time_array.append(predict_time)
        print('Test Data train time : {}\t predict time : {}'.format(train_time, predict_time))
        incorrect = np.count_nonzero(np.subtract(ypredict, y_test))
        acc= 1- incorrect / len(ypredict)
        print('Test Result')
        print(acc)
        accuracy.append(acc)
        if best_acc< acc:
            best_c = c
            best_acc = acc
    print("Train Times = ", train_time_array)
    print("Predict Time = ", predict_time_array)
    print("Accuracy", accuracy)
    print("Best Accuracy", best_acc)

    plt.plot(c_param, accuracy)
    plt.xlabel('Regularization Strength')
    plt.ylabel('Test Accuracy')
    plt.savefig('LogReg.png')
