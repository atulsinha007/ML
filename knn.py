import math
import queue as Q
import load_data as ld
import sys
import test

global testPoint

def distance(x1,x2):
        dis = 0.0;
        for i,j in zip(x1,x2):
            dis += (i-j)*(i-j)
        dis = math.sqrt(dis)
        return dis

class DataPoint:
    def __init__(self, point, type):
        # type refers to the class of the point in classifier
        self.__type =  type
        self.__x = point

    def setDistance(self, point):
        self.distance =  distance(self.__x, point)

    def getPoint(self):
        return self.__x

    def getClass(self):
        return self.__type

    def __lt__(self, other):
        return self.distance < other.distance

    def __eq__(self, other):
        return self.distance == other.distance



def analyse_queue(que):
    count = {}
    while not que.empty():
        p = que.get()
        if not count.get(p.getClass()):
            count[p.getClass()] = 0
        count[p.getClass()] += 1 

    return max([(value,key) for key,value in count.items()])[1]


def main():
    train_x, train_y, test_x, test_y = ld.load_data()
    
    datapoints = [DataPoint(x,y) for x,y in zip(train_x, train_y)]
    k = int(input())

    output = []
    for x,y in zip(test_x, test_y):
        que = Q.PriorityQueue(maxsize=k)
        for p in datapoints:
            p.setDistance(x)

            if que.full():
                q = que.queue[0]
                if p < q:
                    que.get()
                    que.put(p)
            else:
                que.put(p)
        output.append((analyse_queue(que),y))

    print(test.accuracy(output))

if __name__ == '__main__':
    main()

