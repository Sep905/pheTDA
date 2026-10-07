# A Topological Data Analysis Framework for Computational Phenotyping 

pheTDA is a semi-supervised topological data analysis pipeline for discovering patient stratifications. It constructs a Mapper graph from mixed tabular
features, detects graph communities, and uses phenotype-aware graph entropy together with community silhouette to guide a multi-objective Optuna
search.

![img1](figures/framework.png?raw=true)

📍 Results of the original work from AIME 2003<sup>I</sup>, check ```/notebooks_AIME_2023_paper/```;

🌾 Code for the pheTDA application on pediatric celiac disease<sup>II</sup>, check ```/notebooks_AIME_2023_paper/```;

🎯 pheTDA optimized with [Optuna python package](https://optuna.readthedocs.io/en/stable/), check ```/pheTDA/```:

The pipeline can use:

- an initial phenotype;
- a final phenotype; or
- both initial and final phenotypes.

Phenotypes guide hyperparameter selection but must not also be included among the input features used to construct the Mapper graph.


#### Citations:
[I] Albi, G., Gerbasi, A., Chiesa, M., Colombo, G.I., Bellazzi, R., Dagliati, A. (2023). A Topological Data Analysis Framework for Computational Phenotyping. In: Juarez, J.M., Marcos, M., Stiglic, G., Tucker, A. (eds) Artificial Intelligence in Medicine. AIME 2023. Lecture Notes in Computer Science(), vol 13897. Springer, Cham. https://doi.org/10.1007/978-3-031-34344-5_38 

[II] Albi, G., Brembilla, V., Lenzi, E., Maffioletti, S., Medolago, E., Sirtoli, C., Ferramosca, A., Lenti, M. V., Di Sabatino, A., Dagliati, A., & Pala, D. (2026). A New Computational Phenotyping Framework for the Clinical Characterization of Pediatric Celiac Disease. Computer Methods and Programs in Biomedicine, 109648. https://doi.org/10.1016/j.cmpb.2026.109648
