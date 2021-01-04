
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from dataManipulator import loadtraindata, loadtestdata, splitdata
import numpy as np
import time


if __name__ == '__main__':

    ###### Hyper Parameters #######

    alp  = [0.0001, 0.001, 0.01, 0.1, 1]
    layers = [(50,50,50), (50,100,50), (100,), (100,100)]

    #######

    X,y = loadtraindata()
    X_test, y_test = loadtestdata()
    print("Finish loading {} data points")
    Xtrain, ytrain,XVal, yval = splitdata(X,y,8)
    print("Running Neural Network on {} training samples and {} Validation samples".format(len(ytrain), len(yval)))

    Xtrain = np.divide(Xtrain, 255)
    XVal =  np.divide(XVal, 255)
    X_test = np.divide(X_test, 255)
    count = 0
    for l in layers:
        print("Running on {} Layers".format(l))
        train_time_array = []
        predict_time_array = []
        accuracy = []
        best_alpha = 0
        best_acc = -1
        for a in alp:
            print("Running on {} alpha".format(a))

            start = time.time()
            NN = MLPClassifier(hidden_layer_sizes=l, max_iter=400, alpha=a,
                                solver='sgd', tol=1e-4, random_state=1)
            NN.fit(Xtrain, ytrain)
            train_time = time.time() - start
            train_time_array.append(train_time)
            start = time.time()
            predict = NN.predict(XVal)
            predict_time = time.time() - start
            print('Validation Data train time : {}\t predict time : {}'.format(train_time, predict_time))
            incorrect = np.count_nonzero(np.subtract(predict, yval))
            result= 1 - incorrect / len(predict)
            print('Validation Result')
            print(result)
            start = time.time()
            ypredict = NN.predict(X_test)
            predict_time = time.time() - start
            predict_time_array.append(predict_time)
            print('Test Data train time : {}\t predict time : {}'.format(train_time, predict_time))
            incorrect = np.count_nonzero(np.subtract(ypredict, y_test))
            acc= 1- incorrect / len(ypredict)
            print('Test Result')
            print(acc)
            accuracy.append(acc)
            if best_acc< acc:
                best_alpha = a
                best_hl = l
                best_acc = acc
        plt.plot(alp, accuracy)
        plt.xlabel('Regularization Strength')
        plt.ylabel('Test Accuracy')

        s = 'NN_'+str(count)+'_layers.png'
        count+=1
        plt.savefig(s)
        print("Train Times = ", train_time_array)
        print("Predict Time = ", predict_time_array)
        print("Accuracy", accuracy)
        print("Best Accuracy", best_acc)
