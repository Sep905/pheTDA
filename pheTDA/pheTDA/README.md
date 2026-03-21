```
pheTDA/                
├── optuna_pipeline/ 
│   ├── __init__.py
│   └── pipelines.py          ->  defines the pheTDA pipeline as a Class
├── pipeline_objects/
│   ├── __init__.py
│   ├── Lens_function.py      ->  Lens_function as a Class - projection
│   ├── nn.py                 ->  additional definitions for the Lens_function based on neural networks
│   ├── Covering.py           ->  Cover as a Class (with modified KeplerMapper package) - covering
│   └── Partitioning.py       ->  Community decetion and stratification introduciton in the population as a Class - partitioning
├── utils/
│   ├── __init__.py
│   ├── graph_utils.py        ->  graph utils such as entropy computation, stratification inroduction in the population
│   └── prepro.py             ->  distance matrix computation
└── pipeline_run.py           ->  script that executes the pipeline 
```

### To run the pheTDA pipeline with optuna:
- create a subfolder ```./data``` to place the dataset as a .csv file
- specify dataset path, the name of the initial class (and its type), the continue, categorical and binary variables, and the entropy direction (minimize or maximize) and run the pipeline as:

```python pipeline_run.py --dataset_path ./data/dataset.csv --initial_class Y --initial_class_type "categorical" --seed 125 --results_path ./results --continue_features "[]" --categorical_features "[]" --binary_features "[]" --entropy minimize```
