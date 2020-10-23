# MOCF

This repository contains the implementation source code of the following paper:

Feasible Actionable Recourse through Sound Multi-objective Counter-factual Explanations

# Setup
1- Install the following package that contains GCC/g++ compilers and libraries:
```
sudo apt-get install build-essential
```
2- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanras/MOCF
```
3- Create a conda virtual environment:
```
conda create -n MOCF python=3.6
```
4- Activate the conda environment: 
```
conda activate MOCF
```
5- Standing in MOCF directory, install the requirements:
```
pip install -r requirements.txt
```

# Reproducing the results
1- To explain a particular instance using MOCF run:
```
python main.py
```
2- To explain a particular instance using MOCF, CFPrototype, and DiCE methods run:
```
python mocf_cfprototype_dice.py
```
1- To reproduce the base comparison results run:
```
python benchmark_base.py
```
2- To reproduce the sound comparison results run:
```
python benchmark_sound.py
```
3- To reproduce the feasible comparison results run:
```
python benchmark_feasible.py
```
4- To reproduce the sound & feasible comparison results run:
```
python benchmark_sound_feasible.py
```
5- To reproduce the soundness validation results run:
```
python validate_soundness.py
```
