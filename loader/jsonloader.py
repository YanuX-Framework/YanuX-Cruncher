import json
import os


class JsonLoader(object):
    def __init__(self, path):
        self.path = path
        with open(self.path) as json_file:
            self.json_data = json.load(json_file)

    def isdirectory(self):
        return os.path.isdir(self.path)
