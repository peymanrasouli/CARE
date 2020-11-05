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

# Reproducing the results
1- To explain a particular instance using CARE run:
```
python main.py
```
2- To explain a particular instance using CARE, CFPrototype, and DiCE run:
```
python care_cfprototype_dice.py
```
3- To reproduce the CARE performance results run:
```
python care_performance.py
```
4- To reproduce the base comparison results run:
```
python benchmark_base.py
```
5- To reproduce the sound comparison results run:
```
python benchmark_sound.py
```
6- To reproduce the sound+causality comparison results run:
```
python benchmark_sound_causality.py
```
7- To reproduce the sound+causality+actionable  comparison results run:
```
python benchmark_sound_causality_actionable.py
```
8- To reproduce the soundness validation results run:
```
python validate_soundness.py
```
9- To reproduce the action series results run:
```
python mocf_action_series.py
```
