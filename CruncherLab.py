import getopt
import os
import sys

from sklearn import preprocessing

from yanux.cruncher.ml.experiments import *
from yanux.cruncher.model.loader import JsonLoader
from yanux.cruncher.model.wifi import WifiLogs


def main(argv):
    help_message = "CruncherLab.py -i <input_data_directory> -o <output_data_directory> -k <k_neighbors>"
    input_data_directory = ""
    output_data_directory = ""
    k_neighbors = 1
    try:
        opts, args = getopt.getopt(argv, "hi:o:k:", ["input-dir=", "output-dir=", "k-neighbors="])
    except getopt.GetoptError:
        print(help_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(help_message)
            sys.exit()
        elif opt in ("-i", "--input-dir"):
            input_data_directory = arg
        elif opt in ("-o", "--output-dir"):
            output_data_directory = arg
        elif opt in ("-k", "--k-neighbors"):
            k_neighbors = int(arg)
    print("Input Data Directory is:", input_data_directory)
    print("Output Data Directory is", output_data_directory)
    print("K Nearest Neighbors:", k_neighbors)
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

    raw = True
    groupby_mean = True
    groupby_max = True
    groupby_min = True
    data_partials = [0.5, 0.15]
    test_data_partials = [0.5, 0.15]
    filename_prefixes = ["point", "altPoint"]
    subset_locations_values = [0.24]

    print("Generating Training and Test Data...")
    data_scenarios = {}
    test_data_scenarios = {}
    # prepare_full_data_scenarios(wifi_samples, data_scenarios,
    #                             raw=raw,
    #                             groupby_mean=groupby_mean,
    #                             groupby_max=groupby_max,
    #                             groupby_min=groupby_min)
    prepare_full_data_scenarios(wifi_samples, test_data_scenarios,
                                raw=raw,
                                groupby_mean=groupby_mean,
                                groupby_max=groupby_max,
                                groupby_min=groupby_min)

    prepare_partial_data_scenarios(wifi_samples, data_scenarios,
                                   slice_at_the_end=False,
                                   raw=raw,
                                   groupby_mean=groupby_mean,
                                   groupby_max=groupby_max,
                                   groupby_min=groupby_min,
                                   partials=data_partials)
    prepare_partial_data_scenarios(wifi_samples, test_data_scenarios,
                                   slice_at_the_end=True,
                                   raw=raw,
                                   groupby_mean=groupby_mean,
                                   groupby_max=groupby_max,
                                   groupby_min=groupby_min,
                                   partials=test_data_partials)

    #for filename_prefix in filename_prefixes:
    #    prepare_filename_startswith_data_scenarios(wifi_samples, data_scenarios,
    #                                                raw=raw,
    #                                                groupby_mean=groupby_mean,
    #                                                groupby_max=groupby_max,
    #                                                groupby_min=groupby_min,
    #                                                filename_startswith=filename_prefix)
    # for filename_prefix in filename_prefixes:
    #     prepare_filename_startswith_data_scenarios(wifi_samples, test_data_scenarios,
    #                                                raw=raw,
    #                                                groupby_mean=groupby_mean,
    #                                                groupby_max=groupby_max,
    #                                                groupby_min=groupby_min,
    #                                                filename_startswith=filename_prefix)

    #for subset_locations in subset_locations_values:
    #    prepare_full_data_scenarios(subset_wifi_samples_locations(wifi_samples, subset_locations), data_scenarios,
    #                                raw=raw,
    #                                groupby_mean=groupby_mean,
    #                                groupby_max=groupby_max,
    #                                groupby_min=groupby_min,
    #                                scenarios_suffix="subset_locations=" + str(subset_locations))

    # path_direction_aggregated_data_scenarios(wifi_samples, data_scenarios,
    #                                          groupby_mean=groupby_mean,
    #                                          groupby_max=groupby_max,
    #                                          groupby_min=groupby_min)

    save_scenarios(data_scenarios, output_directory=output_data_directory, prefix="train_")
    print("# Data Scenarios: " + str(len(data_scenarios)))
    save_scenarios(test_data_scenarios, output_directory=output_data_directory, prefix="test_")
    print("# Test Scenarios: " + str(len(test_data_scenarios)))

    weights_values = ["uniform", "distance"]
    metric_values = ["euclidean", "manhattan", "chebyshev", "canberra", "braycurtis"]
    nan_filler_values = [-100.0, -100000.0]
    units_values = ["dBm", "mW"]
    scaler_values = {"None": None,
                     "MinMaxScaler": preprocessing.MinMaxScaler(),
                     "StandardScaler": preprocessing.StandardScaler(),
                     "RobustScaler": preprocessing.RobustScaler(),
                     "NormalizerEuclidean": preprocessing.Normalizer(norm="l2"),
                     "NormalizerManhattan": preprocessing.Normalizer(norm="l1")}

    scenarios = []
    scenario_keys = None

    for weights in weights_values:
        for metric in metric_values:
            for nan_filler in nan_filler_values:
                for units in units_values:
                    for scaler_name, scaler in scaler_values.items():
                        for data_scenario, data in data_scenarios.items():
                            for test_data_scenario, test_data in test_data_scenarios.items():
                                if k_neighbors < len(data):
                                    print("train_data =", data_scenario)
                                    print("test_data =", test_data_scenario)
                                    print("train_data_size =", len(data))
                                    print("test_data_size =", len(test_data))
                                    print("algorithm =", "KNeighborsRegressor")
                                    print("n_neighbors =", k_neighbors)
                                    print("weights =", weights)
                                    print("metric =", metric)
                                    print("nan_filler =", nan_filler)
                                    print("units =", units)
                                    print("scaler =", scaler_name)
                                    print("----------------------------------------------------------------")
                                    scenario = collections.OrderedDict([("train_data", data_scenario),
                                                                        ("test_data", test_data_scenario),
                                                                        ("train_data_size", len(data)),
                                                                        ("test_data_size", len(test_data)),
                                                                        ("algorithm", "KNeighborsRegressor"),
                                                                        ("n_neighbors", k_neighbors),
                                                                        ("weights", weights),
                                                                        ("metric", metric),
                                                                        ("nan_filler", nan_filler),
                                                                        ("units", units),
                                                                        ("scaler", scaler_name)])
                                    curr_data = data.fillna(nan_filler)
                                    curr_test_data = test_data.fillna(nan_filler)

                                    curr_data[mac_addresses] = convert_to_units(curr_data[mac_addresses],
                                                                                from_units="dBm",
                                                                                to_units=units)
                                    curr_test_data[mac_addresses] = convert_to_units(curr_test_data[mac_addresses],
                                                                                     from_units="dBm",
                                                                                     to_units=units)
                                    if scaler is not None:
                                        scaler.fit(curr_data[mac_addresses])
                                        curr_data[mac_addresses] = pd.DataFrame(
                                            scaler.transform(curr_data[mac_addresses]),
                                            columns=mac_addresses)
                                        curr_test_data[mac_addresses] = pd.DataFrame(
                                            scaler.transform(curr_test_data[mac_addresses]),
                                            columns=mac_addresses)

                                    scenario.update(experiment_metrics(knn_experiment(curr_data,
                                                                                      mac_addresses,
                                                                                      ["x", "y"],
                                                                                      algorithm="brute",
                                                                                      n_neighbors=k_neighbors,
                                                                                      weights=weights,
                                                                                      metric=metric,
                                                                                      test_data=curr_test_data)))
                                    scenario_keys = scenario.keys()
                                    scenarios.append(scenario)

    metrics = pd.DataFrame(scenarios, columns=scenario_keys)
    metrics.to_csv(output_data_directory + "/metrics-k_neighbors=" + str(k_neighbors) + ".csv")


if __name__ == "__main__":
    main(sys.argv[1:])
