import re
from collections import OrderedDict
from model.location import IndoorLocation


class WifiConnectionInfo(object):
    def __init__(self, connection_info):
        self.bssid = connection_info["bssid"]
        self.detailed_state = connection_info["detailedState"]
        self.dhcp_dns1 = connection_info["dhcpDns1"]
        self.dhcp_dns2 = connection_info["dhcpDns2"]
        self.dhcp_gateway = connection_info["dhcpGateway"]
        self.dhcp_ip_address = connection_info["dhcpIpAddress"]
        self.dhcp_lease_duration = connection_info["dhcpLeaseDuration"]
        self.dhcp_netmask = connection_info["dhcpNetmask"]
        self.ip_address = connection_info["ipAddress"]
        self.link_speed = connection_info["linkSpeed"]
        self.mac_adsress = connection_info["macAddress"]
        self.network_id = connection_info["networkId"]
        self.rssi = connection_info["rssi"]
        self.ssid = connection_info["ssid"]
        self.supplicant_state = connection_info["supplicantState"]
        self.ssid_hidden = connection_info["ssidHidden"]


class SensorEntry(object):
    def __init__(self, sensor_entry):
        self.accuracy = sensor_entry["accuracy"]
        self.sensor_name = sensor_entry["sensorName"]
        self.sensor_type = sensor_entry["sensorType"]
        self.sensor_type_name = sensor_entry["sensorTypeName"]
        self.timestamp = sensor_entry["timestamp"]
        self.values = sensor_entry["values"]


class WifiResult(object):
    def __init__(self, wifi_result):
        self.timestamp = wifi_result["timestamp"]
        self.frequency = wifi_result["frequency"]
        self.mac_address = wifi_result["macAddress"]
        self.signal_strength = wifi_result["signalStrength"]
        self.ssid = wifi_result["ssid"]


class WifiReading(object):
    def __init__(self, filename, session_id, timestamp, wifi_reading):
        self.filename = filename
        self.session_id = session_id
        self.timestamp = timestamp
        self.current_connection_info = WifiConnectionInfo(wifi_reading["connectionInfo"])
        self.sensor_entries = []
        for sensor_entry in wifi_reading["sensorEntries"]:
            self.sensor_entries.append(SensorEntry(sensor_entry))
        self.wifi_results = []
        for wifi_result in wifi_reading["wifiResults"]:
            self.wifi_results.append(WifiResult(wifi_result))


class WifiLogs(object):
    _DEFAULT_FLOOR = 2

    def __init__(self, json_data):
        self._json_data = json_data
        self.locations = {}
        self._load()

    def _load(self):
        for name, log_file in self._json_data.items():
            # Find a better way to get the coordinates so that I don't have to follow a specific filename format.
            # Besides that I really shouldn't hardcode the default floor into the code
            parsed_coord = re.findall("\d+", name)
            coordinates = (int(parsed_coord[0]), int(parsed_coord[1]), WifiLogs._DEFAULT_FLOOR)
            if coordinates not in self.locations:
                self.locations[coordinates] = IndoorLocation(coordinates[0], coordinates[1], coordinates[2])
            location = self.locations[coordinates]
            for session in log_file["sessions"]:
                timestamp = session["timestamp"]
                for entry in session["entries"]:
                    location.wifi_samples.append(WifiReading(log_file["name"], entry["id"], timestamp, entry["reading"]))

    def wifi_samples(self):
        wifi_samples = OrderedDict([("filename", []), ("session_id", []), ("x", []), ("y", []), ("floor", []),
                                    ("timestamp", []), ("mac_address", []), ("signal_strength", [])])
        for coordinates, location in self.locations.items():
            for wifi_reading in location.wifi_samples:
                for wifi_result in wifi_reading.wifi_results:
                    wifi_samples["filename"].append(wifi_reading.filename)
                    wifi_samples["session_id"].append(wifi_reading.session_id)
                    wifi_samples["x"].append(location.x)
                    wifi_samples["y"].append(location.y)
                    wifi_samples["floor"].append(location.floor)
                    wifi_samples["timestamp"].append(wifi_result.timestamp)
                    wifi_samples["mac_address"].append(wifi_result.mac_address)
                    wifi_samples["signal_strength"].append(wifi_result.signal_strength)
        return wifi_samples
