
# coding: utf-8

# # Wi-Fi Fingerprinting

# ## Import modules and set up the environment

# In[1]:

get_ipython().magic('matplotlib inline')

import getopt                                         # C-style parser for command line options
import os                                             # Miscellaneous operating system interfaces

import sys                                            # System-specific parameters and functions
import math                                           # Mathematical functions
import time                                           # Time access and conversions

import numpy as np                                    # NumPy
import scipy as sp                                    # SciPy
import pandas as pd                                   # Pandas
pd.set_option('display.max_rows', 10000)              # Increase Pandas maximum rows display

import matplotlib.pyplot as plt                       # Matplotlib
import seaborn as sns                                 # Seaborn
sns.despine()                                         # Removing spines

from sklearn import preprocessing                     # scikit-learn preprocessing data
from sklearn.grid_search import GridSearchCV          # scikit-learn model selection GridSearchCV
from sklearn.cross_validation import train_test_split # scikit-learn model selection train_test_split
from sklearn.metrics import classification_report     # scikit-learn metrics classification_report


# ### Load the model classes

# A class responsible for loading a JSON file (or all the JSON files in a given directory) into a Python dictionary

# In[2]:

from yanux.cruncher.model.loader import JsonLoader


# A class that takes a set of Python dictionaries containing Wi-Fi logging data loaded from JSON files collected by the YanuX Scavenger Android application

# In[3]:

from yanux.cruncher.model.wifi import WifiLogs


# Some general helper functions

# In[4]:

from yanux.cruncher.ml.experiments import *


# ## Initialize Input & Output Data Directories and other parameters

# In[5]:

input_data_directory = "data"
output_data_directory = "out"

print("Input Data Directory is:", input_data_directory)
print("Output Data Directory is", output_data_directory)


# ### Create the output directory if it doesn't exist

# In[6]:

if not os.path.exists(output_data_directory):
    os.makedirs(output_data_directory)


# ## Load Data from the Input Data Directory

# Load all files from the *data* folder.
# The logs currently placed there were collected using the **Yanux Scavenger** Android application on April 28<sup>th</sup>, 2016 using an LG Nexus 5 running Androdid Marshmallow 6.0.1

# In[7]:

json_loader = JsonLoader(input_data_directory+"/wifi-fingerprints")
wifi_logs = WifiLogs(json_loader.json_data)

## Shuffle data to remove any bias that measuring order may have introduced.
## This is mainly relevant because part of the data will be used for training while the rest will be used for testing.
# wifi_logs.shuffle_samples()


# ## Wi-Fi Readings

# Number of Recorded Samples per Location

# In[8]:

num_samples_per_location = int(len(wifi_logs.wifi_samples()) / len(wifi_logs.locations))
num_samples_per_location


# Store the data into a Pandas Dataframe, in which each Wi-Fi result reading is represented by a single line

# In[9]:

wifi_results_columns = ["filename", "x", "y", "floor", "orientation", "sample_id", "mac_address",
                        "timestamp", "signal_strength"]

wifi_results = pd.DataFrame(wifi_logs.wifi_results(), columns=wifi_results_columns)
wifi_results.to_csv(output_data_directory + "/wifi_results.csv")
# wifi_results


# Identify the unique MAC Addresses present in the recorded data. Each one represents a single Wi-Fi Access Point.

# In[10]:

mac_addresses = wifi_results.mac_address.unique()


# Similarly, store the data into a Pandas Dataframe in which each line represents a single sampling cycle with *n* different readings for each of the Access Points within range. Those readings are stored as columns along each sample.

# In[11]:

wifi_samples_columns = ["filename", "x", "y", "floor", "orientation", "sample_id", "timestamp"]
wifi_samples_columns.extend(mac_addresses)

wifi_samples = pd.DataFrame(wifi_logs.wifi_samples(), columns=wifi_samples_columns)
wifi_samples = wifi_samples.sort_values(["filename", "x", "y", "floor", "sample_id"]).reset_index(drop=True)
wifi_samples.to_csv(output_data_directory + "/wifi_samples.csv")


# ## Data Set Statistics

# Number of Results

# In[12]:

len(wifi_results)


# Number of Unique Mac Addresses

# In[13]:

len(wifi_results.mac_address.unique())


# Signal Strength Mean

# In[14]:

wifi_results.signal_strength.mean()


# Signal Strength Standard Deviation

# In[15]:

wifi_results.signal_strength.std()


# Signal Strength Min

# In[16]:

wifi_results.signal_strength.min()


# Signal Strength Max

# In[17]:

wifi_results.signal_strength.max()


# ### How often has each Access Point been detected

# In[18]:

wifi_results_mac_address_group = wifi_results.groupby("mac_address")
wifi_results_mac_address_group.size().plot(kind="bar")


# In[19]:

wifi_results_mac_address_group.size()


# ### How many times Wi-Fi results were gathered at each location

# In[20]:

wifi_results_coord_group = wifi_results.groupby(["x", "y"])
wifi_results_coord_group.size().plot(kind="bar")


# In[21]:

wifi_results_coord_group.size()


# ### The coordinates of the points where data was captured

# In[22]:

coords = wifi_results[["x","y"]].drop_duplicates().sort_values(by=["x","y"]).reset_index(drop=True)
coords_plot_size = (min(coords["x"].min(),coords["y"].min())-2, max(coords["x"].max(),coords["y"].max())+2)
coords.plot(x="x",y="y", style="o", grid=True, legend=False,
            xlim=coords_plot_size, ylim=coords_plot_size,
            xticks=np.arange(coords_plot_size[0], coords_plot_size[1], 2),
            yticks=np.arange(coords_plot_size[0], coords_plot_size[1], 2)).axis('equal')


# ### Signal Strength Distribution

# In[23]:

wifi_results.hist(column="signal_strength")


# In[ ]:




# ## Generate Train and Test Scenario

# In[24]:

raw = True
groupby_mean = False
groupby_max = False
groupby_min = False
data_partials = [0.35]
test_data_partials = [0.35]
filename_prefixes = ["point", "altPoint"]
subset_locations_values = [0.5]


# In[25]:

print("Generating Training and Test Scenarios...")

data_scenarios = {}
test_data_scenarios = {}

full_data_scenarios = {}
prepare_full_data_scenarios(wifi_samples, full_data_scenarios,
                            raw=raw,
                            groupby_mean=groupby_mean,
                            groupby_max=groupby_max,
                            groupby_min=groupby_min)

full_test_data_scenarios = {}
prepare_full_data_scenarios(wifi_samples, full_test_data_scenarios,
                            raw=raw,
                            groupby_mean=groupby_mean,
                            groupby_max=groupby_max,
                            groupby_min=groupby_min)

data_scenarios.update(full_data_scenarios)
test_data_scenarios.update(full_test_data_scenarios)

partial_data_scenarios = {}
prepare_partial_data_scenarios(wifi_samples, partial_data_scenarios,
                               slice_at_the_end=False,
                               raw=raw,
                               groupby_mean=groupby_mean,
                               groupby_max=groupby_max,
                               groupby_min=groupby_min,
                               partials=data_partials)
partial_test_data_scenarios = {}
prepare_partial_data_scenarios(wifi_samples, partial_test_data_scenarios,
                               slice_at_the_end=True,
                               raw=raw,
                               groupby_mean=groupby_mean,
                               groupby_max=groupby_max,
                               groupby_min=groupby_min,
                               partials=test_data_partials)

data_scenarios.update(partial_data_scenarios)
test_data_scenarios.update(partial_test_data_scenarios)

filename_startswith_data_scenarios = {}
for filename_prefix in filename_prefixes:
    prepare_filename_startswith_data_scenarios(wifi_samples, filename_startswith_data_scenarios,
                                               raw=raw,
                                               groupby_mean=groupby_mean,
                                               groupby_max=groupby_max,
                                               groupby_min=groupby_min,
                                               filename_startswith=filename_prefix)
filename_startswith_test_data_scenarios = {}
for filename_prefix in filename_prefixes:
    prepare_filename_startswith_data_scenarios(wifi_samples, filename_startswith_test_data_scenarios,
                                               raw=raw,
                                               groupby_mean=groupby_mean,
                                               groupby_max=groupby_max,
                                               groupby_min=groupby_min,
                                               filename_startswith=filename_prefix)

data_scenarios.update(filename_startswith_data_scenarios)
test_data_scenarios.update(filename_startswith_test_data_scenarios)

subset_locations_data_scenarios = {}
for subset_locations in subset_locations_values:
    prepare_full_data_scenarios(subset_wifi_samples_locations(wifi_samples, subset_locations),
                                subset_locations_data_scenarios,
                                raw=raw,
                                groupby_mean=groupby_mean,
                                groupby_max=groupby_max,
                                groupby_min=groupby_min,
                                scenarios_suffix="subset_locations=" + str(subset_locations))

data_scenarios.update(subset_locations_data_scenarios)


path_direction_aggregated_data_scenarios = {}
prepare_path_direction_aggregated_data_scenarios(wifi_samples, path_direction_aggregated_data_scenarios,
                                         groupby_mean=groupby_mean,
                                         groupby_max=groupby_max,
                                         groupby_min=groupby_min)

data_scenarios.update(path_direction_aggregated_data_scenarios)

#Data Scenarios using sklearn's train_test_split
loc_coords = ["x", "y", "floor"]
X = wifi_samples[mac_addresses].copy()
y = wifi_samples[loc_coords].copy()
y["label"] = "(" + y.x.map(str) + ", " + y.y.map(str) + ", " + y.floor.map(str) + ")"

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y[loc_coords],
                                                    test_size=0.5,
                                                    stratify=y["label"].values,
                                                    random_state=0)
#X_train = X_train.reset_index(drop=True)
#X_test = X_test.reset_index(drop=True)
#y_train = y_train.reset_index(drop=True)
#y_test = y_test.reset_index(drop=True)

if raw or groupby_mean or groupby_max or groupby_min:
    #train data
    train_split_data = pd.concat([y_train, X_train], axis=1).reset_index(drop=True)
    #test data
    test_split_data = pd.concat([y_test, X_test], axis=1).reset_index(drop=True)
    if raw:
        #train data
        data_scenarios["train_test_split_data"] = train_split_data
        #test data
        test_data_scenarios["train_test_split_data"] = test_split_data
    if groupby_mean or groupby_max or groupby_min:
        #train data
        train_split_groupby_data = train_split_data.groupby(loc_coords, as_index=False)
        #test data
        test_split_groupby_data = test_split_data.groupby(loc_coords, as_index=False)
        if groupby_mean:
            #train data
            train_split_groupby_mean_data = train_split_groupby_data.mean()
            data_scenarios["train_test_split_mean_data"] = train_split_groupby_mean_data
            #test data
            test_split_groupby_mean_data = test_split_groupby_data.mean()
            data_scenarios["train_test_split_mean_data"] = test_split_groupby_mean_data
        if groupby_max:
            #train data
            train_split_groupby_max_data = train_split_groupby_data.max()
            data_scenarios["train_test_split_max_data"] = train_split_groupby_max_data
            #test data
            test_split_groupby_max_data = test_split_groupby_data.max()
            data_scenarios["train_test_split_max_data"] = test_split_groupby_max_data
        if groupby_min:
            #train data
            train_split_groupby_min_data = train_split_groupby_data.min()
            data_scenarios["train_test_split_min_data"] = train_split_groupby_min_data
            #test data
            test_split_groupby_min_data = test_split_groupby_data.min()
            data_scenarios["train_test_split_min_data"] = test_split_groupby_min_data

save_scenarios(data_scenarios, output_directory=output_data_directory, prefix="train_")
print("# Data Scenarios: " + str(len(data_scenarios)))
save_scenarios(test_data_scenarios, output_directory=output_data_directory, prefix="test_")
print("# Test Data Scenarios: " + str(len(test_data_scenarios)))
print("Scenarios Generated!")


# Set a train and test scenario to be used by default when testing.

# In[26]:

default_data_scenario = data_scenarios["train_test_split_data"]
default_test_data_scenario = test_data_scenarios["train_test_split_data"]


# ## Playground

# ### Algorithm
# Test if the different nearest neighbor search algorithms produce different results.

# In[27]:

n_neighbors=5
weights="uniform"
metric="euclidean"
nan_filler=-100
algorithms = ["brute", "kd_tree", "ball_tree"]
leaf_sizes = range(10, 60, 10) 

curr_data = default_data_scenario.fillna(nan_filler)
curr_test_data = default_test_data_scenario.fillna(nan_filler)

# Just a metrics accumulator
metrics = []
for a in algorithms:
    for l in leaf_sizes:
        start_time = time.clock()
        results = knn_experiment(curr_data,
                                 mac_addresses,
                                 ["x", "y"],
                                 algorithm=a,
                                 leaf_size=l,
                                 n_neighbors=n_neighbors,
                                 weights=weights,
                                 metric=metric,
                                 test_data=curr_test_data)
        end_time = time.clock()
        curr_metrics = experiment_metrics(results)
        curr_metrics["algorithm"] = a
        curr_metrics["leaf_size"] = l
        curr_metrics["elapsed_time"] = end_time - start_time
        metrics.append(curr_metrics)

cols = ["algorithm", "leaf_size", "elapsed_time"] + list(curr_metrics.keys())[:-3]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-algorithm.csv")
metrics_table.sort_values(cols[3:])
#metrics_table.sort_values("elapsed_time")


# ### # Neighbors
# Test how the *k* value influences performance metrics

# In[28]:

n_neighbors=range(1,11,2)
weights="uniform"
metric="euclidean"
nan_filler=-100

curr_data = default_data_scenario.fillna(nan_filler)
curr_test_data = default_test_data_scenario.fillna(nan_filler)

# Just a metrics accumulator
metrics = []
for k in n_neighbors:
    curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                     mac_addresses,
                                                     ["x", "y"],
                                                     algorithm="brute",
                                                     n_neighbors=k,
                                                     weights=weights,
                                                     metric=metric,
                                                     test_data=curr_test_data))
    curr_metrics["k"] = k
    metrics.append(curr_metrics)

cols = ["k"] + list(curr_metrics.keys())[:-1]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-n_neighbors.csv")
metrics_table.sort_values(cols[1:])


# ### Weights
# Check whether the neighbors should have the same (*uniform*) or a weighted (*distance*-based) influence in the regression result.

# In[29]:

n_neighbors=range(2,6,1)
weights=["uniform", "distance"]
metric="euclidean"
nan_filler=-100

curr_data = default_data_scenario.fillna(nan_filler)
curr_test_data = default_test_data_scenario.fillna(nan_filler)

# Just a metrics accumulator
metrics = []
for k in n_neighbors:
    for w in weights:
        curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                         mac_addresses,
                                                         ["x", "y"],
                                                         algorithm="brute",
                                                         n_neighbors=k,
                                                         weights=w,
                                                         metric=metric,
                                                         test_data=curr_test_data))
        curr_metrics["k"] = k
        curr_metrics["weights"] = w
        metrics.append(curr_metrics)

cols = ["k","weights"] + list(curr_metrics.keys())[:-2]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-weights.csv")
metrics_table.sort_values(cols[2:])


# ### Metric
# Just test a few different distance metrics to assess if there is a better alternative than the plain old *euclidean* distance. The tested metrics include:
# - Euclidean Distance
#     - sqrt(sum((x - y)^2))
# - Manhattan Distance
#     - sum(|x - y|) 
# - Chebyshev Distance
#     - sum(max(|x - y|))
# - Hamming Distance
#     - N_unequal(x, y) / N_tot
# - Canberra Distance
#     - sum(|x - y| / (|x| + |y|))
# - Braycurtis Similarity
#     - sum(|x - y|) / (sum(|x|) + sum(|y|))
# - S Euclidean Distance
#     - sqrt(sum((x - y)^2 / V))
# - Mahalanobis Distance
#     - sqrt((x - y)' V^-1 (x - y))
# 
# The possible arguments are the following:
# - p = The order of the norm of the difference
# - V = array_like symmetric positive-definite covariance matrix.
# - w = (N,) array_like weight vector.

# In[30]:

n_neighbors=3
weights="uniform"
metric=[
        "euclidean","manhattan", "chebyshev",
        "hamming", "canberra", "braycurtis",
        #"seuclidean", "mahalanobis"
       ]
nan_filler=-100

curr_data = default_data_scenario.fillna(nan_filler)
curr_test_data = default_test_data_scenario.fillna(nan_filler)

# Just a metrics accumulator
metrics = []
for m in metric:
    if metric in ["mahalanobis", "seuclidean"]:
        metric_params = {'V': np.cov(curr_data[mac_addresses])}
    else:
        metric_params = None
    curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                     mac_addresses,
                                                     ["x", "y"],
                                                     algorithm="brute",
                                                     n_neighbors=n_neighbors,
                                                     weights=weights,
                                                     metric=m,
                                                     metric_params=metric_params,
                                                     test_data=curr_test_data))
    curr_metrics["metric"] = m
    metrics.append(curr_metrics)

cols = ["metric"] + list(curr_metrics.keys())[:-1]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-metric.csv")
metrics_table.sort_values(cols[1:])


# ### NaN filler values
# Test which is the signal strength value that should be considered for Access Points that are currently out of range. This is needed as part of the process of computing the distance/similarity between different fingerprints.

# In[31]:

n_neighbors=3
weights="uniform"
metric="euclidean"
nan_filler = [-1000000, -100, 0, 100, 1000000,
              data_scenarios["full_data"][mac_addresses].min().min()-1] 

# Just a metrics accumulator
metrics = []
for nf in nan_filler:
    curr_data = default_data_scenario.fillna(nf)
    curr_test_data = default_test_data_scenario.fillna(nf)
    curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                     mac_addresses,
                                                     ["x", "y"],
                                                     algorithm="brute",
                                                     n_neighbors=n_neighbors,
                                                     weights=weights,
                                                     metric=metric,
                                                     test_data=curr_test_data))
    curr_metrics["nan_filler"] = nf
    metrics.append(curr_metrics)

cols = ["nan_filler"] + list(curr_metrics.keys())[:-1]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-nan_filler.csv")
metrics_table.sort_values(cols[1:])


# ### Units
# - dBm
# - mW

# In[32]:

n_neighbors=3
weights="uniform"
metric="euclidean"
nan_filler=-100

# Just a metrics accumulator
metrics = []

# Use the directly measured dBm values
curr_data = default_data_scenario.fillna(nan_filler)
curr_test_data = default_test_data_scenario.fillna(nan_filler)
curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                 mac_addresses,
                                                 ["x", "y"],
                                                 algorithm="brute",
                                                 n_neighbors=n_neighbors,
                                                 weights=weights,
                                                 metric=metric,
                                                 test_data=curr_test_data))
curr_metrics["units"] = "dBm"
metrics.append(curr_metrics)

# Convert to mW
curr_data[mac_addresses] = convert_to_units(curr_data[mac_addresses], from_units="dBm", to_units="mW")
curr_test_data[mac_addresses] = convert_to_units(curr_test_data[mac_addresses], from_units="dBm", to_units="mW")
curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                 mac_addresses,
                                                 ["x", "y"],
                                                 algorithm="brute",
                                                 n_neighbors=n_neighbors,
                                                 weights=weights,
                                                 metric=metric,
                                                 test_data=curr_test_data))
curr_metrics["units"] = "mW"
metrics.append(curr_metrics)

    
cols = ["units"] + list(curr_metrics.keys())[:-1]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-units.csv")
metrics_table.sort_values(cols[1:])


# ### Scaler
# Test different data scaling and normalization approaches to find out if any of them provides a clear advantage over the others.

# In[33]:

n_neighbors=3
weights="uniform"
metric="euclidean"
nan_filler = -100

scaler_values = {
                    "None": None,
                    "MinMaxScaler": preprocessing.MinMaxScaler(),
                    "StandardScaler": preprocessing.StandardScaler(),
                    "RobustScaler": preprocessing.RobustScaler(),
                    "NormalizerEuclidean": preprocessing.Normalizer(norm="l2"),
                    "NormalizerManhattan": preprocessing.Normalizer(norm="l1")
                }

# Just a metrics accumulator
metrics = []
for scaler_name, scaler in scaler_values.items():
    curr_data = default_data_scenario.fillna(nan_filler)
    curr_test_data = default_test_data_scenario.fillna(nan_filler)
    if scaler is not None:
        scaler.fit(curr_data[mac_addresses])
        curr_data[mac_addresses] = pd.DataFrame(scaler.transform(curr_data[mac_addresses]),
                                                columns=mac_addresses)
        curr_test_data[mac_addresses] = pd.DataFrame(scaler.transform(curr_test_data[mac_addresses]),
                                                     columns=mac_addresses)
    curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                     mac_addresses,
                                                     ["x", "y"],
                                                     algorithm="brute",
                                                     n_neighbors=n_neighbors,
                                                     weights=weights,
                                                     metric=metric,
                                                     metric_params=metric_params,
                                                     test_data=curr_test_data))
    curr_metrics["scaler"] = scaler_name
    metrics.append(curr_metrics)

cols = ["scaler"] + list(curr_metrics.keys())[:-1]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-scaler.csv")
metrics_table.sort_values(cols[1:])


# ### Different Data Processing and Aggregation Scenarios
# Testing different ways of processing offline and online data.

# #### How does the amount of retrieved online data before affects positioning performance while considering the full offline data? And also, how do different agrregation strategies (i.e., mean, maximum and minimum signal strength per location) affect the results?

# In[34]:

n_neighbors=3
weights="uniform"
metric="euclidean"
nan_filler=-100

curr_data_scenarios = {}
curr_data_scenarios.update(full_data_scenarios)

curr_test_data_scenarios = {}
curr_test_data_scenarios.update(full_test_data_scenarios)
curr_test_data_scenarios.update(partial_test_data_scenarios)

# Just a metrics accumulator
metrics = []
for data_scenario_name, data_scenario in curr_data_scenarios.items():
    for test_data_scenario_name, test_data_scenario in curr_test_data_scenarios.items():
        curr_data = data_scenario.fillna(nan_filler)
        curr_test_data = test_data_scenario.fillna(nan_filler)
        curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                         mac_addresses,
                                                         ["x", "y"],
                                                         algorithm="brute",
                                                         n_neighbors=n_neighbors,
                                                         weights=weights,
                                                         metric=metric,
                                                         test_data=curr_test_data))
        curr_metrics["data_scenario"] = data_scenario_name
        curr_metrics["test_data_scenario"] = test_data_scenario_name
        metrics.append(curr_metrics)

cols = ["data_scenario", "test_data_scenario"] + list(curr_metrics.keys())[:-2]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-data_scenarios.csv")
metrics_table.sort_values(cols[2:])


# In[35]:

n_neighbors=3
weights="uniform"
metric="euclidean"
nan_filler=-100

curr_data_scenarios = {}
curr_data_scenarios.update(full_data_scenarios)
curr_data_scenarios.update(partial_data_scenarios)

curr_test_data_scenarios = {}
curr_test_data_scenarios.update(full_test_data_scenarios)

# Just a metrics accumulator
metrics = []
for data_scenario_name, data_scenario in curr_data_scenarios.items():
    for test_data_scenario_name, test_data_scenario in curr_test_data_scenarios.items():
        curr_data = data_scenario.fillna(nan_filler)
        curr_test_data = test_data_scenario.fillna(nan_filler)
        curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                         mac_addresses,
                                                         ["x", "y"],
                                                         algorithm="brute",
                                                         n_neighbors=n_neighbors,
                                                         weights=weights,
                                                         metric=metric,
                                                         test_data=curr_test_data))
        curr_metrics["data_scenario"] = data_scenario_name
        curr_metrics["test_data_scenario"] = test_data_scenario_name
        metrics.append(curr_metrics)

cols = ["data_scenario", "test_data_scenario"] + list(curr_metrics.keys())[:-2]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-data_scenarios.csv")
metrics_table.sort_values(cols[2:])


# #### Does orientation affect the positioning performance?
# Comparison between data collected when moving through the floor plan from left to right, top to bottom, and when moving in the opposite directions. 

# In[36]:

n_neighbors=3
weights="uniform"
metric="euclidean"
nan_filler=-100

curr_data_scenarios = {}
curr_data_scenarios.update(filename_startswith_data_scenarios)

curr_test_data_scenarios = {}
curr_test_data_scenarios.update(filename_startswith_test_data_scenarios)

# Just a metrics accumulator
metrics = []
for data_scenario_name, data_scenario in curr_data_scenarios.items():
    for test_data_scenario_name, test_data_scenario in curr_test_data_scenarios.items():
        curr_data = data_scenario.fillna(nan_filler)
        curr_test_data = test_data_scenario.fillna(nan_filler)
        curr_metrics = experiment_metrics(knn_experiment(curr_data,
                                                         mac_addresses,
                                                         ["x", "y"],
                                                         algorithm="brute",
                                                         n_neighbors=n_neighbors,
                                                         weights=weights,
                                                         metric=metric,
                                                         test_data=curr_test_data))
        curr_metrics["data_scenario"] = data_scenario_name
        curr_metrics["test_data_scenario"] = test_data_scenario_name
        metrics.append(curr_metrics)

cols = ["data_scenario", "test_data_scenario"] + list(curr_metrics.keys())[:-2]
metrics_table = pd.DataFrame(metrics, columns=cols)
metrics_table.to_csv(output_data_directory + "/metrics-data_scenarios.csv")
metrics_table.sort_values(cols[2:])


# ### Grid Search - Searching for estimator parameters

# In[37]:

k_neighbors_values = range(1,11)
weights_values = [
                    "uniform",
                    "distance"
                 ]
metric_values = [
                    "euclidean",
                    "manhattan",
                    "chebyshev",
                    "hamming",
                    "canberra", 
                    "braycurtis"
                ]
algorithm_values = [
                    "auto",
                    "brute",
                    "ball_tree",
                    #"kd_tree"
                   ]
leaf_size_values = range(10, 60, 10)
nan_filler = -100

param_grid = {
                "n_neighbors": list(k_neighbors_values),
                "weights": weights_values,
                "metric": metric_values,
                "algorithm": algorithm_values,
                "leaf_size": list(leaf_size_values)
              }
knn = KNeighborsRegressor()
grid = GridSearchCV(knn, param_grid=param_grid, cv=10, n_jobs=-1, error_score=0,
                    scoring=sklearn.metrics.make_scorer(sklearn.metrics.r2_score,
                                                        greater_is_better=True,
                                                        needs_proba=False,
                                                        needs_threshold=False,
                                                        multioutput="uniform_average"))
grid.fit(X_train.fillna(nan_filler), y_train[["x", "y"]])
print("Best parameters set found on development set:")
print(grid.best_params_)
print("Grid scores on development set:")
for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" %(mean_score, scores.std() * 2, params))


# In[38]:

y_true, y_pred = y_test, grid.predict(X_test.fillna(-100))
#Only applicable to classifiers. I'm currently using regression!!
#print(classification_report(y_true, y_pred))

predictions = pd.DataFrame(y_pred, columns=["x", "y"])
result = y_true.reset_index(drop=True).join(predictions, rsuffix="_predicted")
result["error"] = (predictions[["x", "y"]] - result[["x", "y"]]).apply(np.linalg.norm, axis=1)
metrics = experiment_metrics(result)
pd.DataFrame([metrics], columns=list(metrics.keys()))


# ## Parameter Sweeping

# Initialize some variables with the values of each parameter that is going to be swept.

# In[39]:

k_neighbors_values = range(1,10)
weights_values = ["uniform", "distance"]
metric_values = ["euclidean","manhattan", "chebyshev", "canberra", "braycurtis"]
nan_filler_values = [-100.0, -100000.0]
units_values = ["dBm", "mW"]
scaler_values = {"None": None,
                 "MinMaxScaler": preprocessing.MinMaxScaler(),
                 "StandardScaler": preprocessing.StandardScaler(),
                 "RobustScaler": preprocessing.RobustScaler(),
                 "NormalizerEuclidean": preprocessing.Normalizer(norm="l2"),
                 "NormalizerManhattan": preprocessing.Normalizer(norm="l1")}


# #### Manual Parameter Estimation with for loops
# Do the actual parameter estimation by manually sweeping through each parameter's possible value, while keeping track of the metrics for each parameter combination.

# In[40]:

scenarios = []
scenario_keys = None
# for k_neighbors in k_neighbors_values:
#     for weights in weights_values:
#         for metric in metric_values:
#             for nan_filler in nan_filler_values:
#                 for units in units_values:
#                     for scaler_name, scaler in scaler_values.items():
#                         for data_scenario, data in data_scenarios.items():
#                             for test_data_scenario, test_data in test_data_scenarios.items():
#                                 if k_neighbors < len(data):
# #                                     print("train_data =", data_scenario)
# #                                     print("test_data =", test_data_scenario)
# #                                     print("train_data_size =", len(data))
# #                                     print("test_data_size =", len(test_data))
# #                                     print("algorithm =", "KNeighborsRegressor")
# #                                     print("n_neighbors =", k_neighbors)
# #                                     print("weights =", weights)
# #                                     print("metric =", metric)
# #                                     print("nan_filler =", nan_filler)
# #                                     print("units =", units)
# #                                     print("scaler =", scaler_name)
# #                                     print("----------------------------------------------------------------")
#                                     print(".", end='')
#                                     scenario = collections.OrderedDict([("train_data", data_scenario),
#                                                                         ("test_data", test_data_scenario),
#                                                                         ("train_data_size", len(data)),
#                                                                         ("test_data_size", len(test_data)),
#                                                                         ("algorithm", "KNeighborsRegressor"),
#                                                                         ("n_neighbors", k_neighbors),
#                                                                         ("weights", weights),
#                                                                         ("metric", metric),
#                                                                         ("nan_filler", nan_filler),
#                                                                         ("units", units),
#                                                                         ("scaler", scaler_name)])
#                                     curr_data = data.fillna(nan_filler)
#                                     curr_test_data = test_data.fillna(nan_filler)
#                                     curr_data[mac_addresses] = convert_to_units(curr_data[mac_addresses],
#                                                                                 from_units="dBm",
#                                                                                 to_units=units)
#                                     curr_test_data[mac_addresses] = convert_to_units(curr_test_data[mac_addresses],
#                                                                                      from_units="dBm",
#                                                                                      to_units=units)
#                                     if scaler is not None:
#                                         scaler.fit(curr_data[mac_addresses])
#                                         curr_data[mac_addresses] = pd.DataFrame(scaler.transform(curr_data[mac_addresses]),
#                                                                                 columns=mac_addresses)
#                                         curr_test_data[mac_addresses] = pd.DataFrame(scaler.transform(curr_test_data[mac_addresses]),
#                                                                                      columns=mac_addresses)
#                                     scenario.update(experiment_metrics(knn_experiment(curr_data,
#                                                                                       mac_addresses,
#                                                                                       ["x", "y"],
#                                                                                       algorithm="brute",
#                                                                                       n_neighbors=k_neighbors,
#                                                                                       weights=weights,
#                                                                                       metric=metric,
#                                                                                       test_data=curr_test_data)))
#                                     scenario_keys = scenario.keys()
#                                     scenarios.append(scenario)
print("\n"+str(len(scenarios))+" scenarios have been simulated.")


# Save the metrics to disk for further analysis.

# In[41]:

metrics = pd.DataFrame(scenarios, columns=scenario_keys)
metrics.to_csv(output_data_directory + "/metrics.csv")

