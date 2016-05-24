class IndoorLocation(object):
    def __init__(self, x, y, floor):
        # Local Coordinates (e.g., using a reference grid over a building's floor plan)
        self.x = int(x)
        self.y = int(y)
        self.floor = int(floor)
        # Maybe I should change the nomenclature from readings to samples if it is scientifically more correct
        self.wifi_samples = []

    def __str__(self):
        return "X: "+str(self.x)+" Y: "+str(self.y)+" Floor: "+str(self.floor)
