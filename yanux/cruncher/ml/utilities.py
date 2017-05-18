import pandas as pd 
from sklearn import preprocessing


def normalize(data, mode="rescale"):
    if mode == "rescale":
        scaler = preprocessing.MinMaxScaler().fit(data)
    elif mode == "standardize":
        scaler = preprocessing.StandardScaler().fit(data)
    else:
        raise ValueError("Unsupported normalization mode")

    return pd.DataFrame(scaler.transform(data), columns=data.columns)
 