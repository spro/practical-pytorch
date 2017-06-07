def interpolate(i, v_from, v_to, over):
    return (v_from - v_to) * max(0, (1 - i / over)) + v_to

class SlidingAverage:
    def __init__(self, name, steps=100):
        self.name = name
        self.steps = steps
        self.t = 0
        self.ns = []
        self.avgs = []
    
    def add(self, n):
        self.ns.append(n)
        if len(self.ns) > self.steps:
            self.ns.pop(0)
        self.t += 1
        if self.t % self.steps == 0:
            self.avgs.append(self.value)

    @property
    def value(self):
        if len(self.ns) == 0: return 0
        return sum(self.ns) / len(self.ns)

    def __str__(self):
        return "%s=%.4f" % (self.name, self.value)
    
    def __gt__(self, value): return self.value > value
    def __lt__(self, value): return self.value < value