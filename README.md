# A Topological Data Analysis Framework for Computational Phenotyping 

![img1](figures/framework.png?raw=true)

#### Citation:
Albi, G., Gerbasi, A., Chiesa, M., Colombo, G.I., Bellazzi, R., Dagliati, A. (2023). A Topological Data Analysis Framework for Computational Phenotyping. In: Juarez, J.M., Marcos, M., Stiglic, G., Tucker, A. (eds) Artificial Intelligence in Medicine. AIME 2023. Lecture Notes in Computer Science(), vol 13897. Springer, Cham. https://doi.org/10.1007/978-3-031-34344-5_38 

#### Requirements
- ```requirements.txt``` contains the Python requirements for running the package.
- A tabular dataset made of N rows (patients or sample), M features (clinical features), a class **Y** that define the initial clinical phenotype and an id column **PATIENT_ID** defining the samples id.
- Note that pheTDA leverages [KeplerMapper python package](https://kepler-mapper.scikit-tda.org/en/latest/) for the TDA Mapper implementation.

#### Example: run the pheTDA TDA pipeline
```python
python pheTDA/TDA_Mapper.py --dataset_path "./data/dataset.xlsx" --initial_class "Y" --make_the_class_binary False --control_value 0 --patient_id "PATIENT_ID" --seed 203 --continue_features "['Age','BMI']" --list_lens_functions "['PCA','tSNE']" --n_dimension_projection 2 --perplexities "[5,15,25,35,45]" --learning_rates "[100,300,600,900]" --n_iters "[1500]" --min_dists "[0.25,0.5,0.75,0.9]" --n_neighbors "[5,10,25,50,75]" --resolution "[14,16,18,20,22]" --gain "[0.2,0.3,0.5]" --cluster_method "['DBSCAN','agglomerative_average','agglomerative_single','agglomerative_complete','spectral_clustering','kmedoids']"
``` 

#### Example: run the pheTDA computational phenotyping
```python
python pheTDA/Computational_phenotyping.py --dataset_path "data/dataset.xlsx" --initial_class "Y" --patient_id "PATIENT_ID" --distance_matrix_path "data/distance_matrix.npy" --continue_features "['Age','BMI']" binary_features "['myocardial.infarction']" --n_dimension_projection 2 --seed 203  --projection_lens_path "results/lens_final_model.pkl" --resolution 22 --gain 0.5 --cluster_method "results/cluster_method_final_model.pkl" --colormap "coolwarm" --community_detection_algorithm "Louvain" --list_of_classifiers ["logistic regression","random forest"] --cv_split  5 --flag_remove_duplicate_nodes True
``` 

### Edit: run the pheTDA pipeline while using [Optuna python package](https://optuna.readthedocs.io/en/stable/) to optimize the hyperparameters selection. 

- We use [Pareto optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization) to:
1) **minimize** the fraction of isolated nodes resulting from the TDA Mapper algorithm;
2) **maximize** the modularity of the partition obtained after the community detection;
3) **maximize** the silhouette coefficient after the communities assigment to the patients.

- You need to indicate the seed, the lens and the clustering method. In addition, the path where the dataset is, the path where you would like to have the results and an additional string that indicate which strategy to apply in case of ties during the communities assignment.

```python
python pheTDA/TDA_pipeline_optuna.py --rn 203 --lens "UMAP" --clustering_method "DBSCAN" --dataset_path data/ --results_path results/optuna/ --ties_strategy "node size"
``` 

- You can visualize the optuna results, and choose the configuration of hyperparameters. These can be used to perform the computational phenotyping as in ```Computational_phenotyping.py```
