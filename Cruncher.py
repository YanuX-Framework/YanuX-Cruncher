import numpy as np

from yanux.cruncher.model.loader import JsonLoader
from yanux.cruncher.model.wifi import WifiLogs
from sklearn.cross_validation import train_test_split


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

X = np.array([[1, 2], [3, 4], [4, 6], [7, 8]])
y = np.array(["[0,0]", "[0,0]", "[0,1]", "[0,1]"])
#y = np.array([[0,0], [0,1], [1,0], [1,1]])
#y = np.array([0, 0, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.5,
                                                    stratify=y,
                                                    random_state=0)

print("X_train")
print(X_train)
print("y_train")
print(y_train)
print()
print("X_test")
print(X_test)
print("y_test")
print(y_test)
print("--- THE END ---")
