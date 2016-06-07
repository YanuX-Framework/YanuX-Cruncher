from yanux.cruncher.model.loader import JsonLoader
from yanux.cruncher.model.wifi import WifiLogs

json_loader = JsonLoader('data')
wifi_logs = WifiLogs(json_loader.json_data)

print("# Data Samples")
print(len(json_loader.json_data))

print("# Locations")
print(len(wifi_logs.locations))

print("# Samples")
print(len(wifi_logs.wifi_samples()))

print("# Results")
print(len(wifi_logs.wifi_results()))

print("--- THE END ---")
