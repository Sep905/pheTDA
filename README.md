# A Topological Data Analysis Framework for Computational Phenotyping 

G. Albi, A. Gerbasi, M. Chiesa, G.I. Colombo, R. Bellazzi, and A. Dagliati - accepted at the AIME 2023 conference, June 2023.

### Requirements
- ```requirements.txt``` contains the Python requirements for running the package.
- Patient

## AIME2023 paper supplementary results in ```./CAD_paper_results```:
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

- Figure with the highlighted overall results of the grid search. A) Training set 2D projections for each lenses and B) the graph statistics plotted and highlighted for the second step. 
![img1](figures/img1_highlighted.png?raw=true)

- **Table 3**: classifier models trained using a one-vs-rest binary classification task to predict the patient’s membership to each subgroup. For each model we report the hyperparameters tuned, the range and the best score obtained for each subgroup (in bold if the higher for the subgroup).
