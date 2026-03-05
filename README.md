Contributing to the [HIPE-OCRepair competition 2026](https://hipe-eval.github.io/HIPE-OCRepair-2026/about)  

BL team made up of Valentina Vavassori and Harry Lloyd 
## Project Organization

```
├── LICENSE            <- MIT License
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-HL-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ocrepair and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── uv.lock   <- The uv.lock file for reproducing the analysis environment
│
└── ocrepair   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ocrepair a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


Code scaffold by Cookiecutter Data Science