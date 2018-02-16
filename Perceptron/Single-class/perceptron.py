import numpy as np
import load_data as ld
import test
import sys

def normalize_data(A,mean,sd):              # normalizes and adds 1 in the begining of every row
    l = []
    for i in range(0,len(A)):
        temp = [1]
        for j in range(0,len(A[i])):
            temp.append((A[i][j]-mean)/sd)
        l.append(temp)

    return np.array(l)

def scale_classes(A):               # convert labels to -1 or 1
    m = min(A)
    l = []
    for x in A:
        if x == m:
            l.append(-1)
        else:
            l.append(1)
    return np.array(l)


def initialize(d):
        return np.random.normal(0, 1/np.sqrt(d), d)


class Perceptron():
    def __init__(self, l_rate, batch_size, n_epoch):
        self.__lRate = l_rate
        self.__bSize = batch_size
        self.__epoch = n_epoch
        self.__w = 0
    ###################

    def train(self, X, Y):
        m = len(X)
        d = len(X[0])

        l = self.__lRate
        n = self.__bSize
        e = self.__epoch

        w = initialize(d)
        for t in range(e):
            error = np.array([0.0]*d)
            count = 0
            flag = True

            for x,y in zip(X,Y):
                corr= x*y
                if np.dot(w,corr) < 0.0:
                    error -= corr
                    flag = False

                count += 1
                if count == n:
                    w += l * error/count
                    count = 0
                    error = np.array([0.0]*d)

            if count > 0:
                w += l * error/count

            if flag:                # break if no error found in current epoch i.e early stopping
                break

        self.__w = w;
        print("Training Complete", t)
    ####################

    def test(self, x):
        return np.sign(np.dot(self.__w, x))


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

def main():
    train_x, train_y, test_x, test_y = ld.load_data(sys.argv[1])
    
    train_y = scale_classes(train_y)
    test_y = scale_classes(test_y)
    
    mean = np.mean(train_x)
    sd = np.std(train_y)
    train_x = normalize_data(train_x,mean,sd)
    test_x = normalize_data(test_x,mean,sd)     # test data is normalized using mean and sd of train data

    b,l,e = getParameters()
    model = Perceptron(b,l,e)
    model.train(train_x, train_y)

    output = []
    for x,y in zip(test_x, test_y):
        output.append((model.test(x) , y))

    print(test.accuracy(output))

if __name__ == '__main__':
    main()