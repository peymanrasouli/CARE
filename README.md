# CARE

This repository contains the implementation source code of the following paper:

[CARE: Coherent Actionable Recourse based on Sound Counterfactual Explanations](https://arxiv.org/abs/2108.08197)

Bibtex:

    @misc{rasouli2021care,
          title={CARE: Coherent Actionable Recourse based on Sound Counterfactual Explanations}, 
          author={Peyman Rasouli and Ingrid Chieh Yu},
          year={2021},
          eprint={2108.08197},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

# Setup
1- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanrasouli/CARE
```
2- Install the following package containing GCC/g++ compilers and libraries:
```
sudo apt-get install build-essential
```
3- Create a conda virtual environment:
```
conda create -n CARE python=3.7
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
2- To simultaneously explain a particular instance using CARE, CFPrototype, DiCE, and CERTIFAI run:
```
python care_cfprototype_dice_certifai.py
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
python care_coherency_preservation.py
```

# Reproducing the benchmark results
1- To reproduce the results of CARE<sub>{VALIDITY}</sub> config vs. baselines run:
```
python benchmark_validity.py
```
2- To reproduce the results of CARE<sub>{VALIDITY, SOUNDNESS}</sub> config vs. baselines run:
```
python benchmark_validity_soundness.py
```
3- To reproduce the results of CARE<sub>{VALIDITY, SOUNDNESS, COHERENCY}</sub> config vs. baselines run:
```
python benchmark_validity_soundness_coherency.py
```
4- To reproduce the results of CARE<sub>{VALIDITY, SOUNDNESS, COHERENCY, ACTIONABILITY}</sub> config vs. baselines run:
```
python benchmark_validity_soundness_coherency_actionability.py
```
5- To reproduce the results of coherency-preservation benchmark of CARE vs. baselines run:
```
python benchmark_coherency_preservation.py
```
