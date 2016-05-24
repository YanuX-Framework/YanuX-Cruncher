from model.jsonloader import JsonLoader
from model.wifi import WifiLogs

path = 'data'
json_loader = JsonLoader(path)

print("# Data Samples")
print(len(json_loader.json_data))

wifi_log_stats = WifiLogs(json_loader.json_data)
print("# Unique MAC Addresses")

print(len(wifi_log_stats.mac_addresses))

print("# Locations")
print(len(wifi_log_stats.locations))
print(wifi_log_stats.locations)

