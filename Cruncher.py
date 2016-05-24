from model.jsonloader import JsonLoader
from model.wifi import WifiLogs

path = 'data'
json_loader = JsonLoader(path)

print("# Data Samples")
print(len(json_loader.json_data))

wifi_logs = WifiLogs(json_loader.json_data)
print("# Unique MAC Addresses")

print(len(wifi_logs.mac_addresses))

print("# Locations")
print(len(wifi_logs.locations))
print(wifi_logs.locations)

flat_wifi_readings = wifi_logs.flat_wifi_readings()
print(flat_wifi_readings)
