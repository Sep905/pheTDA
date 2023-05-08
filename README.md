# A Topological Data Analysis Framework for Computational Phenotyping 

G. Albi, A. Gerbasi, M. Chiesa, G.I. Colombo, R. Bellazzi, and A. Dagliati - accepted at the AIME 2023 conference, June 2023.

### Requirements
- ```requirements.txt``` contains the Python requirements for running the package.
- Patient

## AIME2023 paper supplementary results in ```./CAD_paper_results```:
- clinical variables considered in ```./CAD_paper_results/clinical_variables_list.txt``` 
- **Table 1**: population characteristics from CUORE project, divided according to the CAD class. Continue variables are reported as mean (standard deviation) and categorical variables as relative frequency (percentage). Chi-squared/Fisher’s exact test or t-test/Mann-Whitney test are used to assess significant difference between the groups for categorical and continuous variables, respectively. P values are reported in bold if significant(≤ 0.05).

| Feature name, unit of measure | NOATH (n = 287) | ATH (n = 438) | P value |   
| ----------------------------- | --------------- | ------------- | ------- |
| Age, years    | 55 (11.2)     | 61.6 (9.8)    | **<0.001**      |
| Smokers/ex smokers    | 135 (47%)    | 265 (60.5%)    | **<0.001**      |
| Systolic blood pressure, mmHg    | 138.4 (18.1)     | 144.3 (19.2)    | **<0.001**      |
| Total cholesterol, mg/dL    | 197.5 (38.7)     | 193.5 (38.4)    | 0.17      |
| HDL cholesterol, mg/dL    | 63 (17.5)    | 56.9 (15.5)    | **<0.001**     |
| Diabetes mellitus    | 14 (4.9%)    | 57 (13.0 %)    | **<0.001**      |
| Antihypertensives    | 97 (33.8%)    | 250 (57.1 %)    | **<0.001**      |
| CUORE 10-year cardiovascular risk, prob    | 0.07(0.08)     | 0.16 (0.14)    | **<0.001**      |

- **Table 2**: lens functions, hyperparameters, grid search values and the minimum graph entropy obtained for each lens. The optimal score (in bold) indicates the optimal lens resulting from the first step of the grid search.

| Lens function (f) | Hyperparameters (θ’) | Grid search values | Graph entropy H(g) |   
| ------------------| -------------------- | ------------------ | ------------------ |
| PCA               |          -           |      -     |     -      | 0.745              |
| t-SNE             | learning rate<br>perplexity | [300, 600, 900]<br>[15, 25, 35, 45]   | 0.682              |
| UMAP              | minimum distance<br>n° of neighbours | [0.25,0.5,0.75,**0.9**]<br> [5,10,25,**50**,120,150,200]      | **0.657**             |
| UMAP autoencoder  | first hidden layer size<br>n° of hidden layers | [3, 4]<br> [200, 400]     | 0.703            |
| UMAP encoder      | hidden layers size<br>n° of hidden layers | [3, 5]<br> [100, 200]       | 0.713              |
| (CUORE Cardiovascular risk , l2 norm)  |          -           |     -      | 0.756             |

- **Table 3**: Mapper parameters and their grid search values. Values in bold are chosen according to graph statistics.

| Mapper parameters θ | Hyperparameters (θ’) | Grid search values |   
| ------------------- | -------------------- | ------------------ |
| Resolution (r)      |          -           | [14, 16, **18**, 20, 22]|
| Gain (g)      |          -           | [0.2, 0.3, **0.5**, 0.6]|
| Cluster method (C):<br>agglomerative complete-linkage<br>spectral clustering<br>DBSCAN| c<br>n° of clusters (N)<br>n° of clusters (N)<br>epsilon<br>minimum samples | [2,3]


- **Table 3**: classifier models trained using a one-vs-rest binary classification task to predict the patient’s membership to each subgroup. For each model we report the hyperparameters tuned, the range and the best score obtained for each subgroup (in bold if the higher for the subgroup).
