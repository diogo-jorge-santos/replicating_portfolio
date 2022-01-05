# replicating_portfolio

This repository tries to implement a framework to predict hedging errors when hedging a option contract without making assumptions about the contract or strategy used.

-/stable_tools/misc.py -> miscellaneous methods;

-/stable_tools/replicating_fixed_time.py -> base class for replicating fixed time strategies;

-/test_tools/. -> scripts used to test some numba's features and to test some alternative design choices;

-/option_class.py -> implementation of some methods using the base class to test some results from Yet Another Note on the Leland’s Option Hedging Strategy with Transaction Costs Valeri I. Zakamouline

-/main.py -> usage of the methods implemented in option_class.py 

Setup:
- install miniconda/anaconda python distribution.
- To setup an enviorment for this repository use open command line in the repository folder and:
conda env create -f environment.yml

Usage:
- To activate the enviorment and use the scripts:
  - conda activate Master_Thesis_ENV
  - python main.py

Note:
- the results are still slightly off when comparing to the references (even when using large samples and considering the confidence intervals over the obtained results);



  
