import math
import queue as Q
import load_data as ld
import sys
import test
import numpy as np
global testPoint

def distance(x1,x2):
        dis = 0.0;
        for i,j in zip(x1,x2):
            dis += (i-j)*(i-j)
        dis = np.sqrt(dis)
        return dis


def main():

    train_x, train_y, test_x, test_y = ld.load_data()
    dic = {}

    for x, y in zip(train_x, train_y):
        dic[y] = []

    for x, y in zip(train_x, train_y):
        dic[y].append(x)

    dic_centroid = {}

    for x,y in dic.items():
        y = np.asarray(y)
        mean_val = y.mean(axis = 0)
        print(mean_val)
        dic_centroid[x] = mean_val

    print("No. of classes = ", len(dic))

    output = []

    for x,y in zip(test_x, test_y):
        min_dist = sys.maxsize
        for a,b in dic_centroid.items():
            b = np.asarray(b)
            dist = distance(x, b)
            if dist < min_dist:
                min_dist = dist
                class_predicted = a

        output.append((class_predicted, y))

    print(test.accuracy(output))

if __name__ == '__main__':
    main()
