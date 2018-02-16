import numpy as np
import load_data as ld
import test
import sys

class Model():
    def __init__(self, l_rate, batch_size, n_epoch):
        self.__lRate = l_rate
        self.__bSize = batch_size
        self.__epoch = n_epoch
        self.__wList = 0
        self.__nClass = 0
    ###################

    def train(self, X, Y):
        m = len(X)
        d = len(X[0])
        n_class = len(set(Y))         # no. of classes

        l = self.__lRate
        n = self.__bSize
        e = self.__epoch

        w_list = [np.zeros(d)] * n_class

        for t in range(e):
            corr = [np.zeros(d)] * n_class
            count = 0

            for x,y in zip(X,Y):
                y1 = max([(np.dot(w_list[i],x),i) for i in range(n_class)])[1]
                if y1 != y:
                    corr[y1] = corr[y1] - x
                    corr[y] = corr[y] + x
                count += 1

                if count == n:
                    for i in range(n_class):
                        w_list[i] = w_list[i] + l*corr[i]
                    # print(w_list)
                    # print(corr)
                    corr = [np.zeros(d)] * n_class
                    count = 0

            for i in range(n_class):
                w_list[i] = w_list[i] + l*corr[i]

        self.__wList = w_list;
        self.__nClass = n_class
        print("Training Complete")
    ####################

    def test(self, x):
        return max([(np.dot(self.__wList[i],x),i) for i in range(self.__nClass)])[1]


def getParameters():
    print("Enter batch size (1 for online learning, 0 for full-batch):")
    b= int(input())
    if b == 0:
        b = 99999999

    print("Enter learning rate:")
    l = float(input())

    print("Enter Max epochs:")
    e = int(input())

    return l,b,e

def main(l,b,e):
    train_x, train_y, test_x, test_y = ld.load_data(sys.argv[1])
    
    model = Model(l,b,e)
    model.train(train_x, train_y)

    output = []
    for x,y in zip(test_x, test_y):
        output.append((model.test(x) , y))

    print("Accuracy = ", test.accuracy(output))
    # return test.accuracy(output)


if __name__ == '__main__':
    l,b,e = getParameters()
    main(l,b,e)