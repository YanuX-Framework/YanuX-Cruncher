from model.jsonloader import JsonLoader
from model.wifi import WifiLogs

json_loader = JsonLoader('data')

print("# Data Samples")
print(len(json_loader.json_data))

wifi_logs = WifiLogs(json_loader.json_data)

print("# Locations")
print(len(wifi_logs.locations))

wifi_samples = wifi_logs.wifi_samples()
print(wifi_samples)
