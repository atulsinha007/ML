import queue as Q

class Skill(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        print ('New Level:', description)
        return
    def __lt__(self, other):
        return cmp(self.priority, other.priority)

q = Q.PriorityQueue()

q.put(Skill(5, 'Proficient'))
q.put(Skill(10, 'Expert'))
q.put(Skill(1, 'Novice'))

while not q.empty():
    next_level = q.get()
    print ('Processing level:', next_level.description)
