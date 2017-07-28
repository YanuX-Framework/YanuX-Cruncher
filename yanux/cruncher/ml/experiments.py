import collections
import random

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline


def experiment_metrics(result):
    base_error = pd.DataFrame(0, index=np.arange(len(result)), columns=["error"])
    metrics = collections.OrderedDict([("mean_absolute_error", result["error"].mean()),
                                       ("std_dev_distance_error", result["error"].std()),
                                       ("mean_squared_error", sklearn.metrics.mean_squared_error(base_error,
                                                                                                 result["error"])),
                                       ("percentile_25", result["error"].quantile(q=0.25)),
                                       ("percentile_50", result["error"].quantile(q=0.50)),
                                       ("percentile_75", result["error"].quantile(q=0.75)),
                                       ("percentile_90", result["error"].quantile(q=0.90)),
                                       ("percentile_95", result["error"].quantile(q=0.95)),
                                       ("min", result["error"].min()),
                                       ("max", result["error"].max())])
    return metrics


def knn_experiment(data, test_data, train_cols, coord_cols,
                   scaler=None, n_neighbors=5, weights='uniform', algorithm='auto',
                   leaf_size=30, p=2, metric='minkowski',
                   metric_params=None, n_jobs=1):
    result = None
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                              leaf_size=leaf_size, p=p, metric=metric,
                              metric_params=metric_params, n_jobs=n_jobs)
    if scaler is not None:
        estimator = make_pipeline(scaler, knn)
    else:
        estimator = knn
    
    locations = data.groupby(coord_cols).indices.keys()
    for coords in locations:
        train_data = data[(data[coord_cols[0]] != coords[0]) |
                          (data[coord_cols[1]] != coords[1])].reset_index(drop=True)
        
        target_values = test_data[(test_data[coord_cols[0]] == coords[0]) &
                                  (test_data[coord_cols[1]] == coords[1])].reset_index(drop=True)
        
        estimator.fit(train_data[train_cols], train_data[coord_cols])
        predictions = pd.DataFrame(estimator.predict(target_values[train_cols]), columns=coord_cols)
        curr_result = target_values[coord_cols].join(predictions, rsuffix="_predicted")
        error = pd.DataFrame((predictions[coord_cols] - curr_result[coord_cols]).apply(np.linalg.norm, axis=1),
                             columns=["error"])
        curr_result = pd.concat([curr_result, error], axis=1)
        result = pd.concat([result, curr_result])

    return result


def knn_experiment_cv(data, cross_validation, train_cols, coord_cols,    
                      scaler=None, n_neighbors=5, weights='uniform',
                      algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                      metric_params=None, n_jobs=1):    
    result = None
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                              leaf_size=leaf_size, p=p, metric=metric,
                              metric_params=metric_params, n_jobs=n_jobs)
    if scaler is not None:
        estimator = make_pipeline(scaler, knn)
    else:
        estimator = knn
        
    X = data[train_cols]
    y = data[coord_cols]
    
    predictions = pd.DataFrame(cross_val_predict(estimator, X, y, cv=cross_validation), columns=coord_cols)
    result = y.join(predictions, rsuffix="_predicted")
    
    error = pd.DataFrame((predictions[coord_cols] - result[coord_cols]).apply(np.linalg.norm, axis=1), columns=["error"])
    result = pd.concat([result, error], axis=1)
    
    return result