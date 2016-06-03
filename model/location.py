import random


class IndoorLocation(object):
    def __init__(self, x, y, floor):
        # Local Coordinates (e.g., using a reference grid over a building's floor plan)
        self.x = int(x)
        self.y = int(y)
        self.floor = int(floor)
        self.wifi_samples = []

    def __str__(self):
        return "X: "+str(self.x)+" Y: "+str(self.y)+" Floor: "+str(self.floor)

    def shuffle_samples(self):
        random.shuffle(self.wifi_samples)
