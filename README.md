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
git clone https://github.com/peymanras/CARE
```
3- Create a conda virtual environment:
```
conda create -n CARE python=3.6
```
4- Activate the conda environment: 
```
conda activate CARE
```
5- Standing in CARE directory, install the requirements:
```
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
1- To reproduce the CARE performance results run:
```
python care_performance.py
```
2- To reproduce the CARE soundness results run:
```
python care_soundness.py
```
3- To reproduce the CARE action series results run:
```
python care_action_series.py
```
4- To reproduce the CARE causality preservation results run:
```
python care_causality_preservation.py
```

# Reproducing the benchmark results
1- To reproduce the base benchmark results run:
```
python benchmark_base.py
```
2- To reproduce the sound benchmark results run:
```
python benchmark_sound.py
```
3- To reproduce the sound+causality benchmark results run:
```
python benchmark_sound_causality.py
```
4- To reproduce the sound+causality+actionable  benchmark results run:
```
python benchmark_sound_causality_actionable.py
```
