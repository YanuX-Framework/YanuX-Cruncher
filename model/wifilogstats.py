import re
from model.location import Location


class WifiLogStats(object):
    def __init__(self, json_data):
        self._json_data = json_data
        self.mac_addresses = self._mac_addresses()
        self.locations = self._locations()
        
    def _mac_addresses(self):
        mac_addresses = set()
        for name, log_file in self._json_data.items():
            for session in log_file["sessions"]:
                for entry in session["entries"]:
                    mac_addresses.add(entry["reading"]["connectionInfo"]["bssid"])
                    for wifiResults in entry["reading"]["wifiResults"]:
                        mac_addresses.add(wifiResults["macAddress"])
        return mac_addresses
    
    def _locations(self):
        locations = []
        for name, log_file in self._json_data.items():
            coord = re.findall("\d+", name)
            location = Location(coord[0], coord[1])
            locations.append(location)
        return locations
