# CARE

This repository contains the implementation source code of the following paper:

CARE: Causality-preserved Actionable Recourse based on Sound Counterfactual Explanations

# Setup
1- Install the following package that contains GCC/g++ compilers and libraries:
```
sudo apt-get install build-essential
```
2- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanrasouli/CARE
```
3- Create a conda virtual environment:
```
conda create -n CARE python=3.7
```
4- Activate the conda environment: 
```
conda activate CARE
```
5- Standing in CARE directory, install the baselines and requirements:
```
pip install -r baselines.txt
pip install -r requirements.txt
```

# Explaining instances
1- To explain a particular instance using CARE run:
```
python main.py
```
2- To explain a particular instance using CARE, CFPrototype, and DiCE run:
```
python care_cfprototype_dice.py
```

# Reproducing the validation results
1- To reproduce the results of module effect validation run:
```
python care_module_effect.py
```
2- To reproduce the results of soundness validation run:
```
python care_soundness.py
```
3- To reproduce the results of action series validation run:
```
python care_action_series.py
```
4- To reproduce the results of causality-preservation validation run:
```
python care_causality_preservation.py
```

# Reproducing the benchmark results
1- To reproduce the results of CARE with {validity} config vs. CFPrototype and DiCE run:
```
python benchmark_validity.py
```
2- To reproduce the results of CARE with {validity+soundness} config vs. CFPrototype and DiCE run:
```
python benchmark_validity_soundness.py
```
3- To reproduce the results of CARE with {validity+soundness+causality} config vs. CFPrototype and DiCE run:
```
python benchmark_validity_soundness_causality.py
```
4- To reproduce the results of CARE with {validity+soundness+causality+actionability} config vs. CFPrototype and DiCE run:
```
python benchmark_validity_soundness_causality_actionability.py
```
5- To reproduce the results of causality-preservation benchmark of CARE vs. CFPrototype and DiCE run:
```
python benchmark_causality_preservation.py
```
