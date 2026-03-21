import os
import warnings
from scipy.sparse import SparseEfficiencyWarning
from optuna.exceptions import ExperimentalWarning

# Silence Optuna's Experimental warnings for TPESampler
warnings.filterwarnings("ignore", category=ExperimentalWarning)
# Prevent the actual OpenMP crash and KMeans memory leak - Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Silence the persistent threadpoolctl RuntimeWarning (using regex)
warnings.filterwarnings(
    action="ignore", 
    message=".*Found Intel OpenMP.*", 
    category=RuntimeWarning
)

# Silence the Isomap "disconnected graph" UserWarning
warnings.filterwarnings(
    action="ignore", 
    message=".*number of connected components.*", 
    category=UserWarning
)

# Silence the SciPy "csr_matrix is expensive" SparseEfficiencyWarning
warnings.filterwarnings(
    action="ignore", 
    category=SparseEfficiencyWarning
)

warnings.filterwarnings(
    action="ignore", 
    message="using precomputed metric; inverse_transform will be unavailable", 
    category=UserWarning
)
warnings.filterwarnings("ignore", message=".*stable_cumsum.*", category=FutureWarning)

import pandas as pd
import argparse
import optuna
from optuna_pipeline import SemiSupervised_TDA_pipeline
from utils import distance_matrix_computation
import math
import os
from tqdm import tqdm
import numpy as np

def main(args):

    continue_features = eval(args.continue_features)
    categorical_features = eval(args.categorical_features)
    binary_features = eval(args.binary_features)

    # read the dataset
    dataset = pd.read_csv(args.dataset_path)
    dataset['ID'] = range(len(dataset)) 

    if not(os.path.exists(args.results_path)):
        os.mkdir(args.results_path )

    if not(os.path.exists(args.results_path + "/" + str(args.seed) )):
        os.mkdir(args.results_path + "/" + str(args.seed) )

    # compute the distance matrix
    if not(os.path.exists(args.results_path + "/distance_matrix.npy")):
        distance_matrix, dataset_X_features = distance_matrix_computation(dataset['ID'], 
                                                                        dataset[args.initial_class], 
                                                                        dataset, 
                                                                        continue_features,
                                                                        categorical_features, 
                                                                        binary_features,
                                                                        args.results_path)
    else:
        distance_matrix = np.load(args.results_path + "/distance_matrix.npy")
        dataset_X_features = pd.read_csv(args.results_path + "/dataset_preprocessed.csv")



    # create pipeline
    objective = SemiSupervised_TDA_pipeline(distance_matrix,
                                                dataset_X_features, 
                                                args.seed, 
                                                dataset,
                                                dataset['ID'], 
                                                dataset[args.initial_class], 
                                                args.initial_class_type,
                                                args.results_path + "/" + str(args.seed), 
                                                args.entropy)


    # create optuna study with directions for the object to optimize
    entropy_dir = optuna.study.StudyDirection.MINIMIZE if args.entropy == "minimize" else optuna.study.StudyDirection.MAXIMIZE
            

    #sampler_optuna = optuna.samplers.NSGAIISampler(seed = args.seed, population_size =100)
    sampler_optuna = optuna.samplers.TPESampler(seed = args.seed, multivariate=True, n_startup_trials= 500, constant_liar=True)

    study = optuna.create_study(directions=[ entropy_dir, optuna.study.StudyDirection.MAXIMIZE], sampler = sampler_optuna)

    target_n_trials = 1000
    batch_size = 10
    n_batches = math.ceil(target_n_trials / batch_size)  
    
    print("Running Optuna optimization:")
    with tqdm(total=target_n_trials, desc="Optuna Trials") as pbar:
        while len(study.trials) < target_n_trials:
            remaining = target_n_trials - len(study.trials)
            current_batch_size = min(batch_size, remaining)
                
            # STEP 1: Ask → GET FrozenTrials DIRECTLY
            frozen_trials = []
            for _ in range(current_batch_size):
                frozen_trial = study.ask()  
                frozen_trials.append(frozen_trial)
                
            # STEP 2: Sequential evaluation 
            values_list = []
            for frozen_trial in frozen_trials:
                values = objective(frozen_trial) 
                values_list.append(values)
                
            # STEP 3: Tell using trial.number
            for frozen_trial, values in zip(frozen_trials, values_list):
                study.tell(frozen_trial, values)  
                
            pbar.update(current_batch_size)

    study.trials_dataframe().to_excel(args.results_path + "/" + str(args.seed) + "/" + "df_results.xlsx",index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TDA')

    parser.add_argument('--dataset_path', type=str, default = "./data/dataset.xlsx")
    parser.add_argument('--initial_class', type=str, default = "Y")
    parser.add_argument('--initial_class_type', type=str, default = "categorical")
    parser.add_argument('--seed', type=int, default=203)
    parser.add_argument('--results_path', type=str, default="")

    #features 
    parser.add_argument('--continue_features',type=str)
    parser.add_argument('--categorical_features',type=str)
    parser.add_argument('--binary_features',type=str)

    parser.add_argument('--entropy', type=str, default="")

    args = parser.parse_args()    
    main(args)