# A Topological Data Analysis Framework for Computational Phenotyping 

![img1](figures/framework.png?raw=true)

#### Citation:
Albi, G., Gerbasi, A., Chiesa, M., Colombo, G.I., Bellazzi, R., Dagliati, A. (2023). A Topological Data Analysis Framework for Computational Phenotyping. In: Juarez, J.M., Marcos, M., Stiglic, G., Tucker, A. (eds) Artificial Intelligence in Medicine. AIME 2023. Lecture Notes in Computer Science(), vol 13897. Springer, Cham. https://doi.org/10.1007/978-3-031-34344-5_38 

#### Requirements
- ```requirements.txt``` contains the Python requirements for running the package.
- A tabular dataset made of N rows (patients or sample), M features (clinical features), a class **Y** that define the initial clinical phenotype and an id column **PATIENT_ID** defining the samples id.
- Note that pheTDA leverages [KeplerMapper python package](https://kepler-mapper.scikit-tda.org/en/latest/) for the TDA Mapper implementation.

#### To explore the results of the original work from AIME 2023, check ```/notebooks_AIME_2023_paper/```

#### To run the pipeline, check ```/pheTDA/```:
- ```/pheTDA_notebook.ipynb``` is the updated notebook with the overall pipeline
- ```/TDA_pipeline_optuna.py``` to run the pipeline with using  [Optuna python package](https://optuna.readthedocs.io/en/stable/) to optimize the hyperparameters selection
  
- We use [Pareto optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization) to:
1) **minimize** the fraction of isolated nodes resulting from the TDA Mapper algorithm;
2) **maximize** the modularity of the partition obtained after the community detection;
3) **maximize** the silhouette coefficient after the communities assigment to the patients.

```python
python pheTDA/TDA_pipeline_optuna.py --rn 203 --lens "UMAP" --clustering_method "DBSCAN" --dataset_path data/ --results_path results/optuna/ --ties_strategy "node size"
``` 

- You can visualize the optuna results, and choose the configuration of hyperparameters. These can be used to perform the computational phenotyping as in ```Computational_phenotyping.py```
