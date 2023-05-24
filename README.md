# A Topological Data Analysis Framework for Computational Phenotyping 

G. Albi, A. Gerbasi, M. Chiesa, G.I. Colombo, R. Bellazzi, and A. Dagliati - accepted at the AIME 2023 conference, June 2023.

### Requirements
- ```requirements.txt``` contains the Python requirements for running the package.
- A tabular dataset made of N rows (patients or sample), M features (clinical features), a binary class **Y** that define the initial clinical phenotype and an id column **PATIENT_ID** defining the samples id.

## Example: run the pheTDA TDA pipeline
```python
python pheTDA/TDA_Mapper.py --dataset_path "../data/dataset.xlsx" --binary_class "Y" --patient_id "PATIENT_ID" --seed 203 --test_set_split_proportion 0.3 --continue_features ["Age","BMI"] --list_lens_functions ["PCA","tSNE","UMAP"] --n_dimension_projection 2 --perplexities list( np.arange(15,55,10)) --learning_rates list( np.arange(300,1000,300)) --n_iters list(np.array([1500])) --min_dists list(np.array([0.25,0.5,0.75,0.9])) --n_neighbors list( np.array([5,10,25,50,120,150,200])) --resolution  list( np.array([14, 16, 18, 20, 22])) --gain list( np.array([0.2, 0.3, 0.5, 0.6]))
``` 

## Example: run the pheTDA computational phenotyping
```python
python pheTDA/Computational_phenotyping.py --trainingset_path "data/trainingset.npy" --testset_path "data/testgset.npy" --binary_class "Y" '--id_paz' "PATIENT_ID" --distance_matrix_path "data/trainingset_distance_matrix.npy" --n_dimension_projection 2 --seed 203  --projection_lens umap.UMAP(n_components =2 , random_state= 203, n_neighbors= 50, min_dist=0.9) --resolution 18 --gain 0.5 --colormap "coolwarm" --community_detection_algorithm "Greedy modularity" --list_of_classifiers ["logistic regression","random forest","XGBoost"] --cv_split  5
``` 

## Paper supplementary results in ```./CAD_paper_results```:
- clinical variables considered in ```./CAD_paper_results/clinical_variables_list.txt``` 

- Results from the **first step of the grid search**. For each row we report the lens functions, hyperparameters and their values, and the minimum graph entropy obtained for each lens. The score (in bold) indicates the optimal lens resulting from the first step of the grid search.

| Lens function (f) | Hyperparameters (θ’) | Grid search values | Graph entropy H(g) |   
| ------------------| -------------------- | ------------------ | ------------------ |
| PCA               |          -           |      -     |     -      | 0.745              |
| t-SNE             | learning rate<br>perplexity | [300, 600, 900]<br>[15, 25, 35, 45]   | 0.682              |
| UMAP              | minimum distance<br>n° of neighbours | [0.25,0.5,0.75,**0.9**]<br> [5,10,25,**50**,120,150,200]      | **0.657**             |
| UMAP autoencoder  | first hidden layer size<br>n° of hidden layers | [3, 4]<br> [200, 400]     | 0.703            |
| UMAP encoder      | hidden layers size<br>n° of hidden layers | [3, 5]<br> [100, 200]       | 0.713              |

- Results from the **second step of the grid search**. For each row we report the Mapper parameters, their relative hyperparameters and their values. Values in bold are chosen according to graph statistics.

| Mapper parameters θ | Hyperparameters (θ’) | Grid search values |   
| ------------------- | -------------------- | ------------------ |
| Resolution (r)      |          -           | [14, 16, **18**, 20, 22]|
| Gain (g)      |          -           | [0.2, 0.3, **0.5**, 0.6]|
| Cluster method (C): |                      |                   |
| agglomerative complete-linkage<br>spectral clustering<br>DBSCAN|<br>n° of clusters (N)<br>n° of clusters (N)<br>epsilon<br>minimum samples | <br>[2,3]<br>[2,3]<br>[0.2, 0.3, **0.5**]<br>[**2**, 4]|

- Figure with the highlighted **overall results of the grid search**. A) Training set 2D projections for each lenses and B) the graph statistics plotted and highlighted for the second step. 

![img1](figures/img1_highlighted.png?raw=true)

- Results from the **computational phenotyping**. Classifier models trained using a one-vs-rest binary classification task to predict the patient’s membership to each subgroup. For each model we report the hyperparameters tuned, the range and the best score (mean and ± accuracy) obtained for each subgroup (in bold if the higher for the subgroup).

| Model | Hyperparameters and values | α' | β' | γ' | δ' | ε' |
| ----- | -------------------------- | -- | -- | -- | -- | -- |
| EN logistic regression | λ_1 = [0.25 0.5 0.75]<br>λ_2 = [0.001 0.01, 0.1, 1, 10] | **0.76±0.08** | **0.93±0.03** | **0.99±0.01** | **0.93±0.02** | **0.96±0.02** |
| Random forest | maximum tree depth = [1, 3, 5]<br>minimum samples to split  = [2, 5, 10]<br>minimum samples in a leaf = [1, 5]<br>n° of estimators = [100, 200, 300]|  0.60±0.08 | 0.79±0.04 | **0.98±0.01** | 0.88±0.03 | 0.96±0.01 |
| XGBoost | gamma = [0, 0.1, 0.2, 0.3]<br>learning rate = [0.1, 0.25, 0.5]<br>maximum depth  = [1, 3, 5]<br>n° of estimators = [100, 200, 300]|  0.56±0.08 | 0.89±0.03 | 0.98±0.01 | 0.91±0.03 | **0.96±0.01** |
