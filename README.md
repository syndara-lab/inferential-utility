# **The Real Deal Behind the Artificial Appeal: Inferential Utility of Tabular Synthetic Data**

Code to reproduce results in "The Real Deal Behind the Artificial Appeal: Inferential Utility of Tabular Synthetic Data", presented during the 40th Conference on Uncertainty in Artificial Intelligence (2024) and available from https://arxiv.org/pdf/2312.07837.

In this work, we highlight the importance of inferential utility and provide empirical evidence against naive inference from synthetic data, whereby synthetic data are treated as if they were actually observed. Before publishing synthetic data, it is essential to develop statistical inference tools for such data. By means of a simulation study, we show that the rate of false-positive findings (type 1 error) will be unacceptably high, even when the estimates are unbiased. Despite the use of a previously proposed correction factor, this problem persists for deep generative models, in part due to slower convergence of estimators and resulting underestimation of the true standard error. We further demonstrate our findings through a case study.

## Experiments
The following class and helper files are included: 
- utils/custom_bayesian.py: class to train Bayesian Network with DAG pre-specification (using synthcity and pgmpy backend)
- utils/custom_ctgan.py: class to train CTGAN (using sdv backend)
- utils/custom_synthpop.py: class to train Synthpop (using synthcity and R's synthpop backend)
- utils/custom_tvae.py: class to train TVAE (using sdv backend)
- utils/disease.py: simulate low-dimensional tabular toy data sampled from an arbitrary ground truth population
- utils/eval.py: functions to calculate and plot inferential utility metrics
- utils/tuning.py: functions to optimize hyperparameters

Hyperparameter optimization study:
- hpo/hyperparam_optuna.py: hyperparameter optimization study for CTGAN and TVAE (using optuna backend) 
- hpo/hyperparam_eval.py: evaluation of the top hyperparameters (tuned for n=500) in n=50 and n=5000 sample sizes
- hpo/hyperparam_optuna.ipynb: notebook containing results of the hyperparameter optimization study

Simulation study: 
- sim_generate.py: sample original data and generate synthetic version(s) using different generative models
- sim_evaluate.py: calculate inferential utility metrics of original and synthetic datasets
- sim_output.ipynb: notebook containing all output (figures and tables) presented in paper

Case study:
- adult_generate.py: sample original data and generate synthetic version(s) from Adult Census Income dataset using CTGAN and Synthpop
- adult_evaluate.py: calculate inferential utility metrics of original and synthetic datasets
- adult_output.ipynb: notebook containing all output (figures and tables) presented in paper

## Cite
If our paper or code helped you in your own research, please cite our work as:

```
@inproceedings{decruyenaere2024synthetic,
  title={The Real Deal Behind the Artificial Appeal: Inferential Utility of Tabular Synthetic Data},
  author={Decruyenaere, Alexander and Dehaene, Heidelinde and Rabaey, Paloma and Polet, Christiaan and Decruyenaere, Johan and Vansteelandt, Stijn and Demeester, Thomas},
  year={2024},
  organization={40th Conference on Uncertainty in Artificial Intelligence}
}
```
