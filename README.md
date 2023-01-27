# `cempy`

Coarsened Exact Matching for causal inference with Python

# Introduction

Matching is a technique of drawing causal inference from observational data. Unlike randomized control trials, which is a gold standard for causal inference in medical fields, there are many situations where randomization to treatment and control groups are not possible. Instead, observed data are used as they are available and observations in treatment and controls are matched to determine the causal effect. The groups are formed naturally and exact matching performs (one to one, one to many, many to many) matching on the coarsened features. 

For details of coarsened exact matching, please see the references below

# Installation

For local installation, clone the repository and run the following commands

```bash
git clone https://github.com/raheems/cempy.git
```

Then open a Jupyter notebook and run this
```python
!pip install -e .
```

# How to use `cempy`

Once installed, you can start importing the package and load some data

```python
import pandas as pd
from cempy import cem

# Load data
ll = pd.read_csv('lelonde.csv')

# Run CEM
f = cem.CEM(data=ll, trt = 'treated', 
            matching_on=['age', 'education', 'black', 'married', 'nodegree', 're74',
                         're75', 'hispanic', 'u74', 'u75', 'q1'], 
            metric=['re78'], 
            exclude = [])

# Get the metric summary
f.meric_summary()

# Get the dataframe with weights 
f.get_weighted_data()
```

See the examples.ipynb file in the repository for a demonstration with some outputs.

# Related software

- [`cem` R package](https://cran.r-project.org/web/packages/cem/vignettes/cem.pdf)

# Further reading

- [CEM: Coarsened Exact Matching Explained](https://medium.com/@devmotivation/cem-coarsened-exact-matching-explained-7f4d64acc5ef)
