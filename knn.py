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
        dis = sqrt(dis)
        return dis

def DataPoint:
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

    def __cmp__(self, other):
        return cmp(distance(self.distance, other.distance))


def analyse_queue(self, que):
    count = {}
    while !que.empty():
        p = que.get()
        if !count.get(p.getClass()):
            count[p.getClass()] = 0
        count[p.getClass()] += 1 
        maxx = -1
        keyy = 0
        for key, val in count.items():
            if(count[key] > maxx):
                keyy = key
                maxx = count[key]

        return max([(value,key) for key,value in count.items()])[1]


def main():
    train_x, train_y, test_x, test_y = ld.load_data(sys.argv[1])
    datapoints = [DataPoint(x,y) for x,y in zip(train_x, train_y)]
    k = int(input())

    outputs = []
    for x,y in zip(test_x, test_y):
        que = Q.Priorityque(maxsize=k)
        for p in datapoints:
            p.setDistance(x)

            if que.full():
                q = que.queue[0]
                if p.distance < q.distance:
                    que.get()
                    q.put(p)
        output.append((analyse_queue(que),y))

    print(test.accuracy(output))

if __name__ == '__main__':
    main()

