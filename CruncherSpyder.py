import getopt
import os
import sys

from sklearn import preprocessing

from yanux.cruncher.model.loader import JsonLoader
from yanux.cruncher.model.wifi import WifiLogs
from yanux.cruncher.ml.experiments import *

input_data_directory = "data"
output_data_directory = "out"

print("Input Data Directory is:", input_data_directory)
print("Output Data Directory is", output_data_directory)

if not os.path.exists(output_data_directory):
    os.makedirs(output_data_directory)
    
json_loader = JsonLoader(input_data_directory)
wifi_logs = WifiLogs(json_loader.json_data)

wifi_results_columns = ["filename", "x", "y", "floor", "orientation", "sample_id", "mac_address",
                        "timestamp", "signal_strength"]

wifi_results = pd.DataFrame(wifi_logs.wifi_results(), columns=wifi_results_columns)
wifi_results.to_csv(output_data_directory + "/wifi_results.csv")

mac_addresses = wifi_results.mac_address.unique()

wifi_samples_columns = ["filename", "x", "y", "floor", "orientation", "sample_id", "timestamp"]
wifi_samples_columns.extend(mac_addresses)

wifi_samples = pd.DataFrame(wifi_logs.wifi_samples(), columns=wifi_samples_columns)
wifi_samples = wifi_samples.sort_values(["filename", "x", "y", "floor", "sample_id"]).reset_index(drop=True)
wifi_samples.to_csv(output_data_directory + "/wifi_samples.csv")

# -----------------------------------------------------------------------------

n_neighbors=1
weights="distance"
metric="euclidean"
nan_filler = -100

curr_data = wifi_samples
curr_test_data = wifi_samples

curr_data = curr_data.fillna(nan_filler)
curr_test_data = curr_test_data.fillna(nan_filler)

metrics = []
results = []

curr_result = knn_experiment(curr_data,
                             mac_addresses,
                             ["x", "y"],
                             algorithm="brute",
                             n_neighbors=n_neighbors,
                             weights=weights,
                             metric=metric,
                             test_data=curr_test_data)
curr_metrics = experiment_metrics(curr_result)
results.append(curr_result)
curr_metrics["desc"] = "dBm"
metrics.append(curr_metrics)

curr_data[mac_addresses] = 10 ** (curr_data[mac_addresses] / 10)
curr_test_data[mac_addresses] = 10 ** (curr_test_data[mac_addresses] / 10)
curr_result = knn_experiment(curr_data,
                             mac_addresses,
                             ["x", "y"],
                             algorithm="brute",
                             n_neighbors=n_neighbors,
                             weights=weights,
                             metric=metric,
                             test_data=curr_test_data)
curr_metrics = experiment_metrics(curr_result)
results.append(curr_result)
curr_metrics["desc"] = "mW"
metrics.append(curr_metrics)

# Save the results
for i, result in enumerate(results):
    result.to_csv(output_data_directory+"/result-"+str(i)+".csv")
    
cols = list(curr_metrics.keys())[-1:]
cols.extend(list(curr_metrics.keys())[:-1])
metrics_table = pd.DataFrame(metrics, columns=cols)
