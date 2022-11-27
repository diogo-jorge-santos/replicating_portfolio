# replicating_portfolio


In this repository it is implemented a framework to predict and study the hedging costs and error of a derivative product when dynamically hedging a derivative contract.

-/stable_tools/misc.py -> miscellaneous methods and constants;

-/stable_tools/replicating_barrier.py -> base class to implement barrier options;

-/stable_tools/replicating_vol.py -> base class to implement european type options with stocastic volatility;

-/stable_tools/replicating_seperate.py -> base class for implement european type options with constant volatility;

-/stable_tools/class_barrier_call.py and class_barrier_put.py-> class definition for Up and out calls and Down and In puts.

-/stable_tools/class_vol_heston -> class definition for european call using Heston model;

-/stable_tools/class_seperate-> class definition for european call using Black-Shoules formula;

-main_*.py -> scipts used to get the results presented in the Thesis document.


Setup:
- install miniconda/anaconda python distribution.
- To setup an enviorment to use this repository: conda env create -f environment.yml
- For the code be able to log the computed results use: python setup.py
    


Usage:
- To activate the enviorment and use the scripts to generate the results for each topic:
  - conda activate Master_Thesis_ENV
  - python main_*.py



  
