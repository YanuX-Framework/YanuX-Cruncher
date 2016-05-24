class IndoorLocation(object):
    def __init__(self, x, y, floor):
        # Local Coordinates (e.g., using a reference grid over a building's floor plan)
        self.x = x
        self.y = y
        self.floor = floor
        self.wifi_readings = []

    def __str__(self):
        return "X: "+str(self.x)+" Y: "+str(self.y)+" Floor: "+str(self.floor)