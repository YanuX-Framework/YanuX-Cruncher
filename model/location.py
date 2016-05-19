class Location(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.fingerprints = []

    def __str__(self):
        return "X:"+self.x+" Y:"+self.y
