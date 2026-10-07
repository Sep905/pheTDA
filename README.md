# A Topological Data Analysis Framework for Computational Phenotyping 

pheTDA is a semi-supervised topological data analysis pipeline for discovering patient stratifications. It constructs a Mapper graph from mixed tabular
features, detects graph communities, and uses phenotype-aware graph entropy together with community silhouette to guide a multi-objective Optuna
search.

![img1](figures/framework.png?raw=true)

To explore the results of the original work from AIME 2023, check ```/notebooks_AIME_2023_paper/```

To use [Optuna python package](https://optuna.readthedocs.io/en/stable/) for hyperparameters optimization, check ```/pheTDA/```. 

The pipeline can use:

- an initial phenotype;
- a final phenotype; or
- both initial and final phenotypes.

Phenotypes guide hyperparameter selection but must not also be included among the input features used to construct the Mapper graph.


#### Citation:
Albi, G., Gerbasi, A., Chiesa, M., Colombo, G.I., Bellazzi, R., Dagliati, A. (2023). A Topological Data Analysis Framework for Computational Phenotyping. In: Juarez, J.M., Marcos, M., Stiglic, G., Tucker, A. (eds) Artificial Intelligence in Medicine. AIME 2023. Lecture Notes in Computer Science(), vol 13897. Springer, Cham. https://doi.org/10.1007/978-3-031-34344-5_38 
