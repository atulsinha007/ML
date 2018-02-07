import load_data as ld
import test
import sys
import math

class Type(object):

    def __init__(self, id):
        self.__id = id
        self.__count = 0
        self.__centroid = []

    def addPoint(self, point):
        if self.__centroid == []:
            self.__centroid = [x for x in point]
            self.__count = 1

        else:
            for i in range(0,len(point)):
                self.__centroid[i] = (self.__centroid[i]*self.__count + point[i])/(self.__count+1)
                self.__count += 1

    def getDistance(self, point):
        dis = 0.0;
        for i,j in zip(self.__centroid,point):
            dis += (i-j)*(i-j)
        dis = math.sqrt(dis)
        return dis

    def getId():
        return self.__id

    def getSize():
        return self.__count;

    def getCentroid():
        return self.__centroid;


def main():
    train_x, train_y, test_x, test_y = ld.load_data(sys.argv[1])
    types = {}

    #training
    for x,y in zip(train_x, train_y):
        if types.get(y) == None:
            types[y] = Type(y)
        types[y].addPoint(x)
    #end

    #testing
    output = []
    for x,y in zip(test_x, test_y):
        ans = min([(val.getDistance(x),key) for key,val in types.items()])[1]
        output.append((ans,y))

    print(test.accuracy(output))


if __name__ == '__main__':
    main()
        
