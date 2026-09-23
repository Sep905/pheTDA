
# A New Computational Phenotyping Framework for the Clinical Characterization of Pediatric Celiac Disease
<p align="center">
<img src="graphical_abstract.PNG" alt="Graphical Abstract" width="720">
</p>

- Celiac Disease (CD) is a common autoimmune disease, with complex and heterogeneous clinical manifestations that makes it difficult to diagnose and characterize to date, especially in the pediatric case.
- To address this problem, a TDA-based computational phenotyping framework (pheTDA) was applied to a large multicentric pediatric celiac disease cohort.
- Seven novel pediatric celiac disease sub-phenotypes were identified, capturing clinical heterogeneity beyond Oslo definitions. Sub-phenotypes were quantitatively and qualitatively characterized using machine learning, statistics, and topological enrichment.
- TDA-derived sub-phenotypes showed lower intra-group heterogeneity compared with standard clinical classifications. Plus, machine learning models predicted TDA-based sub-phenotypes more accurately than the currently used Oslo categories.

## :file_folder: Repository layout
```
pediatric_celiac_disease/      
  Data_exploration_part1.ipynb          # dataset inspection and preprocessing
  Data_exploration_part2.ipynb          # dataset inspection and preprocessing
  TDA_mapper.ipynb                      # to run the grid search for Mapper
  Computational_phenotyping.ipynb       # to run the TDA Mapper pipeline 
  Clustering_Analysis.ipynb             # analysis with traditional clustering for comparison
```

## 📚 Citation
```bibtex
@article{albi2026phetdaceliac,
title = {A New Computational Phenotyping Framework for the Clinical Characterization of Pediatric Celiac Disease},
journal = {Computer Methods and Programs in Biomedicine},
pages = {109648},
year = {2026},
issn = {0169-2607},
doi = {https://doi.org/10.1016/j.cmpb.2026.109648},
url = {https://www.sciencedirect.com/science/article/pii/S0169260726003974},
author = {Giuseppe Albi and Valentina Brembilla and Erika Lenzi and Simone Maffioletti and Emanuele Medolago and Chiara Sirtoli and Antonio Ferramosca and Marco Vincenzo Lenti and Antonio Di Sabatino and Arianna Dagliati and Daniele Pala},
keywords = {Computational Phenotyping, Pediatric Celiac Disease, Topological Data Analysis, TDA Mapper},
}
```
