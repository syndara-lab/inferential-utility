"""Simulation of toy data."""

import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.genmod.generalized_linear_model import GLM
import warnings

def sample_disease(nsamples, seed=2023) -> pd.DataFrame:
    """
    Sample from imaginary disease
    """
    
    # Fix seed
    rng = np.random.default_rng(seed)
    
    # Age (normal distribution)
    age = rng.normal(loc=50, scale=10, size=nsamples)
    
    # Stage (proportional odds linear regression model) 
    stage_tags = ['I', 'II', 'III', 'IV']
    stage_intercepts = [2, 3, 4] # intercepts of stages are spaced evenly
    stage_beta_age = -0.05 # a negative value for beta means that with increasing values for x, the odds increase of being more than a given value k -> easier to exceed intercept 
    stage_logodds_1, stage_logodds_2, stage_logodds_3 = np.array(stage_intercepts).reshape(len(stage_intercepts),1) + np.vstack([stage_beta_age*age]*len(stage_intercepts)) # get logodds from age 
    stage_cumprob_1, stage_cumprob_2, stage_cumprob_3 = expit([stage_logodds_1, stage_logodds_2, stage_logodds_3]) # cumulative probability of exceeding each of the intercepts
    stage_probs = np.stack((stage_cumprob_1, stage_cumprob_2-stage_cumprob_1, stage_cumprob_3-stage_cumprob_2, 1-stage_cumprob_3), axis=1) # transform cumulative probability of exceeding intercept into probability of being in a certain stage, shape (nsamples, 4)
    stage = np.array([rng.choice(stage_tags, size=1, p=stage_prob) for stage_prob in stage_probs]).flatten() # sample from stage probabilities (categorical distribution)
    
    # Biomarker (gamma distribution)
    biomarker_intercept = 4
    biomarker_beta_stage = [0, -1, -2, -3]
    stage_tag_to_biomarker_beta = np.vectorize(dict(zip(stage_tags, biomarker_beta_stage)).get, otypes=[float]) # map stage tag to beta value
    biomarker_betas = biomarker_intercept + stage_tag_to_biomarker_beta(stage) # higher stage = lower beta
    biomarker_expected = 1/biomarker_betas # E(Y) is inverse of summation of betas (including intercept)
    biomarker_shape = 25
    biomarker_scale = biomarker_expected/biomarker_shape 
    biomarker = rng.gamma(shape=biomarker_shape, scale=biomarker_scale, size=nsamples) # higher stage = higher scale = distr more skewed to the right (biomarker likely to be higher)
    
    # Therapy (binary)
    p_therapy = 0.5 # 50% chance of receiving therapy
    therapy = rng.choice([False, True], size=nsamples, p=[1-p_therapy, p_therapy])
    
    # Death (binary)
    death_intercept = -3
    death_beta_age = 0.05
    death_beta_stage = [0, 0.5, 1, 1.5]
    stage_tag_to_death_beta = np.vectorize(dict(zip(stage_tags, death_beta_stage)).get, otypes=[float]) # map stage tag to beta value
    death_beta_therapy = -0.5
    death_logodds = death_intercept + death_beta_age*age + stage_tag_to_death_beta(stage) + death_beta_therapy*therapy
    death_prob = expit(death_logodds)
    death = np.array([rng.choice([False, True], p=[1-dp, dp], size=1) for dp in death_prob]).flatten()
    
    # Aggregate in dataframe
    data = pd.DataFrame({'age': age, 'stage': stage, 'biomarker': biomarker, 'therapy': therapy, 'death': death})
    
    return data

def ground_truth(nsamples=1000000) -> dict:
    """
    Population parameters
    """
    
    # Use large sample estimate to numerically approximate population parameter (for analytically intractable estimators)
    # draw large sample
    large_sample = sample_disease(nsamples, seed=2023) 
    large_sample['stage'] = pd.Categorical(large_sample['stage'], categories=['I', 'II', 'III', 'IV'], ordered=True) # to use 'stage' as outcome in OrderedModel 
    large_sample_dummies = pd.get_dummies(large_sample) # to use 'stage' dummies as predictor in GLM
    large_sample = large_sample.merge(large_sample_dummies) # merge dummies
    # fit regression models
    large_sample_polr = OrderedModel(large_sample['stage'], large_sample[['age']].astype(float), distr='logit').fit(method='bfgs', disp=False)
    large_sample_logr = GLM(large_sample['death'], sm.add_constant(large_sample[['age', 'stage_II', 'stage_III', 'stage_IV', 'therapy']].astype(float)),
                            family=sm.families.Binomial(link=sm.families.links.Logit())).fit()
    with warnings.catch_warnings(): 
        # ignore warning on line 307 of statsmodels/genmod/generalized_linear_model.py: 
        # 'DomainWarning: The InversePower link function does not respect the domain of the Gamma family.'
        warnings.filterwarnings("ignore", lineno=307)
        large_sample_gamr = GLM(large_sample['biomarker'], sm.add_constant(large_sample[['stage_II', 'stage_III', 'stage_IV']].astype(float)),
                                family=sm.families.Gamma(sm.families.links.InversePower())).fit()

    # Population parameter
    ground_truth = {'age_mean': 50,
                    'age_sd': 10,
                    'age_var': 10**2,
                    'biomarker_mean': np.mean(large_sample['biomarker']), # gamma: shape*scale
                    'biomarker_sd': np.std(large_sample['biomarker'], ddof=1), # gamma: sqrt(shape*scale**2)
                    'biomarker_var': np.var(large_sample['biomarker']), # gamma: shape*scale**2
                    'therapy_prop': 0.5,
                    'stage_I_prop': np.mean(large_sample['stage_I']),
                    'stage_II_prop': np.mean(large_sample['stage_II']),
                    'stage_III_prop': np.mean(large_sample['stage_III']),
                    'stage_IV_prop': np.mean(large_sample['stage_IV']),
                    'death_prop': np.mean(large_sample['death']),
                    'stage_age_polr': -0.05,
                    'biomarker_stage_II_gamr': -1,
                    'biomarker_stage_III_gamr': -2,
                    'biomarker_stage_IV_gamr': -3,
                    'death_age_logr': 0.05,
                    'death_stage_II_logr': 0.5,
                    'death_stage_III_logr': 1,
                    'death_stage_IV_logr': 1.5,
                    'death_therapy_logr': -0.5}
    
    # Rescale empirical SE with asymptotic variance (constant c) for plotting
    # SE = c*n^{-a} <-> log(SE) = log(c)-a*log(n) <-> log(SE/c) = -a*log(n)
    unit_rescale = {'age_mean': ground_truth['age_sd'],
                    'biomarker_mean': ground_truth['biomarker_sd'], 
                    'therapy_prop': np.sqrt(ground_truth['therapy_prop']*(1-ground_truth['therapy_prop'])), 
                    'stage_I_prop': np.sqrt(ground_truth['stage_I_prop']*(1-ground_truth['stage_I_prop'])), 
                    'stage_II_prop': np.sqrt(ground_truth['stage_II_prop']*(1-ground_truth['stage_II_prop'])), 
                    'stage_III_prop': np.sqrt(ground_truth['stage_III_prop']*(1-ground_truth['stage_III_prop'])), 
                    'stage_IV_prop': np.sqrt(ground_truth['stage_IV_prop']*(1-ground_truth['stage_IV_prop'])), 
                    'death_prop': np.sqrt(ground_truth['death_prop']*(1-ground_truth['death_prop'])), 
                    'stage_age_polr': np.sqrt(large_sample_polr.cov_params().loc['age', 'age'])*np.sqrt(nsamples),
                    'biomarker_stage_II_gamr': np.sqrt(large_sample_gamr.cov_params().loc['stage_II', 'stage_II'])*np.sqrt(nsamples), 
                    'biomarker_stage_III_gamr': np.sqrt(large_sample_gamr.cov_params().loc['stage_III', 'stage_III'])*np.sqrt(nsamples),
                    'biomarker_stage_IV_gamr': np.sqrt(large_sample_gamr.cov_params().loc['stage_IV', 'stage_IV'])*np.sqrt(nsamples),
                    'death_age_logr': np.sqrt(large_sample_logr.cov_params().loc['age', 'age'])*np.sqrt(nsamples), 
                    'death_stage_II_logr': np.sqrt(large_sample_logr.cov_params().loc['stage_II', 'stage_II'])*np.sqrt(nsamples), 
                    'death_stage_III_logr': np.sqrt(large_sample_logr.cov_params().loc['stage_III', 'stage_III'])*np.sqrt(nsamples),
                    'death_stage_IV_logr': np.sqrt(large_sample_logr.cov_params().loc['stage_IV', 'stage_IV'])*np.sqrt(nsamples),
                    'death_therapy_logr': np.sqrt(large_sample_logr.cov_params().loc['therapy', 'therapy'])*np.sqrt(nsamples)}
    
    return ground_truth, unit_rescale

if __name__ == "__main__":
    
    # Presets
    n_samples = 20
    seed = 2023
    
    # Sample toy data
    data = sample_disease(n_samples, seed=seed)
    print('Imaginary disease (original data)')
    print('\nhead\n', data.head(10))
    print('\ndtypes\n', data.dtypes)
    print('\ndescriptives\n', data.describe(include='all')) 
    
    # Population parameters and rescaling constants
    data_gt, data_rescale = ground_truth(10000)
    print('\npopulation parameters\n', data_gt)
    print('\nasymptotic variance\n', data_rescale)