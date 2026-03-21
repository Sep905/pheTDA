import gower as gw
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances, euclidean_distances
from sklearn import preprocessing
import numpy as np

def distance_matrix_computation(sample_id, Y_class, dataset, continue_features, categorical_features, binary_features, results_path):
    """
    Function for distance matrix calculation, according to the features' type.
    Distance matrix will be saved at ./distance_matrix.npy
    INPUT:
        - sample_id:            (pandas Series) dataset's column indicating the samples IDs
        - Y_class:              (pandas Series) dataset's column indicating the initial class
        - dataset:              (pandas DataFrame) dataset for which calculate the distance
        - continue_features:    (list) list containing the dataset's numerical features
        - categorical_features: (list) list containing the categorical features (>2 levels)
        - binary_features:      (list) list containing the binary features (2 levels)
    OUTPUT:
        - distance_matrix:      (numpy ndarray) features distance matrix
    """
    
    
    # filter the dataset in order to delete the IDs and the output columns
    dataset_features = dataset.loc[:, ~dataset.columns.isin([sample_id.name, Y_class.name])]

    # determine which types of features are present
    has_categorical = len(categorical_features) > 0
    has_binary = len(binary_features) > 0
    has_continuous = len(continue_features) > 0
    
    # Case 1: only categorical and/or binary variables -> Jaccard distance
    if (has_categorical or has_binary) and not has_continuous:
        print("only categorical and/or binary variables -> Jaccard distance")
        
        if has_binary:
            df_binary = dataset_features[binary_features]
            df_binary_dummies = pd.get_dummies(df_binary.astype(str), drop_first=True, dtype=int)
        else:
            df_binary_dummies = pd.DataFrame()
        
        if has_categorical:
            df_categorical = dataset_features[categorical_features]
            df_categorical_dummies = pd.get_dummies(df_categorical.astype(str), drop_first=False, dtype=int)
        else:
            df_categorical_dummies = pd.DataFrame()
        
        df_encoded = pd.concat([df_binary_dummies, df_categorical_dummies], axis=1)
        distance_matrix = pairwise_distances(df_encoded.values, metric="jaccard")
        dataset_to_return = df_encoded
    
    # Case 2: only numerical variables -> Euclidean distance (as per your code call)
    elif has_continuous and not has_categorical and not has_binary:
        print("only numerical variables -> euclidean distance")
        
        X = dataset_features[continue_features].values
        min_max_scaler = preprocessing.StandardScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        dataset_scaled = pd.DataFrame(X_scaled, columns=continue_features)
        
        distance_matrix = euclidean_distances(dataset_scaled, dataset_scaled)
        dataset_to_return = dataset_scaled
    
    # Case 3: mixed features -> Gower distance
    else:
        print("mixed features (categorical/binary and numerical) -> Gower distance")
        
        if has_binary:
            df_binary = dataset_features[binary_features]
            df_binary_dummies = pd.get_dummies(df_binary.astype(str), drop_first=True, dtype=int)
        else:
            df_binary_dummies = pd.DataFrame()
        
        if has_categorical:
            df_categorical = dataset_features[categorical_features]
            df_categorical_dummies = pd.get_dummies(df_categorical.astype(str), drop_first=False, dtype=int)
        else:
            df_categorical_dummies = pd.DataFrame()
        
        df_continuous = dataset_features[continue_features].astype(float)
        df_mixed = pd.concat([df_continuous, df_binary_dummies, df_categorical_dummies], axis=1)
        
        cat_features = ([False] * len(continue_features) + 
                        [True] * (len(df_binary_dummies.columns) + len(df_categorical_dummies.columns)))
        
        distance_matrix = gw.gower_matrix(df_mixed, cat_features=cat_features)
        dataset_to_return = df_mixed
    
    # save the distance matrix as .npy file
    np.save(results_path + "/distance_matrix.npy", distance_matrix, allow_pickle=False)
    dataset_to_return.to_csv(results_path + "/dataset_preprocessed.csv", index=False)
    
    return distance_matrix, dataset_to_return