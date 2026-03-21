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
