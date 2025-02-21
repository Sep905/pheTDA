# A Topological Data Analysis Framework for Computational Phenotyping 

![img1](figures/framework.png?raw=true)

:blue_book:
Albi, G., Gerbasi, A., Chiesa, M., Colombo, G.I., Bellazzi, R., Dagliati, A. (2023). A Topological Data Analysis Framework for Computational Phenotyping. In: Juarez, J.M., Marcos, M., Stiglic, G., Tucker, A. (eds) Artificial Intelligence in Medicine. AIME 2023. Lecture Notes in Computer Science(), vol 13897. Springer, Cham. https://doi.org/10.1007/978-3-031-34344-5_38 

### :wrench: Requirements
- ```requirements.txt``` contains the Python requirements for running the package.
- A tabular dataset made of N rows (patients or sample), M features (clinical features), a class **Y** that define the initial clinical phenotype and an id column **PATIENT_ID** defining the samples id.
- Note that pheTDA leverages [KeplerMapper python package](https://kepler-mapper.scikit-tda.org/en/latest/) for the TDA Mapper implementation.

### Example: run the pheTDA TDA pipeline
```python
python pheTDA/TDA_Mapper.py --dataset_path "../data/dataset.xlsx" --binary_class "Y" --patient_id "PATIENT_ID" --seed 203 --test_set_split_proportion 0.3 --continue_features ["Age","BMI"] --list_lens_functions ["PCA","tSNE","UMAP"] --n_dimension_projection 2 --perplexities list( np.arange(15,55,10)) --learning_rates list( np.arange(300,1000,300)) --n_iters list(np.array([1500])) --min_dists list(np.array([0.25,0.5,0.75,0.9])) --n_neighbors list( np.array([5,10,25,50,120,150,200])) --resolution  list( np.array([14, 16, 18, 20, 22])) --gain list( np.array([0.2, 0.3, 0.5, 0.6]))
``` 

### Example: run the pheTDA computational phenotyping
```python
python pheTDA/Computational_phenotyping.py --trainingset_path "data/trainingset.npy" --testset_path "data/testgset.npy" --binary_class "Y" '--id_paz' "PATIENT_ID" --distance_matrix_path "data/trainingset_distance_matrix.npy" --n_dimension_projection 2 --seed 203  --projection_lens umap.UMAP(n_components =2 , random_state= 203, n_neighbors= 50, min_dist=0.9) --resolution 18 --gain 0.5 --colormap "coolwarm" --community_detection_algorithm "Greedy modularity" --list_of_classifiers ["logistic regression","random forest","XGBoost"] --cv_split  5
``` 

## :dart: Edit: run the pheTDA pipeline while using Optuna python package to optimize the hyperparameters selection. 

- We use Pareto optimization to:
1) minimize the fraction of isolated nodes;
2) maximize the modularity after community detection
3) maximize the silhouette coefficient after the communities assigment to the patients.

- You need to indicate the seed, the lens and the clustering method. In addition, the path where the dataset is, the path where you would like to have the results and an additional string that indicate which strategy to apply in case of ties during the communities assignment.

```python
python pheTDA/TDA_pipeline_optuna.py --rn 203 --lens "UMAP" --clustering_method "DBSCAN" --dataset_path data/ --results_path results/optuna/ --ties_strategy "node size"
``` 

- You can visualize the optuna results, and choose the configuration of hyperparameters. These can be used to perform the computational phenotyping as in ```Computational_phenotyping.py```
