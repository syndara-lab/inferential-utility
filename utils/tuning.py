import numpy as np
import pandas as pd
import optuna
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.generic.plugin_ctgan import CTGANPlugin
from synthcity.plugins.generic.plugin_tvae import TVAEPlugin
from sklearn.model_selection import KFold
from utils.custom_ctgan import CTGAN
from utils.custom_tvae import TVAE
from utils.disease import sample_disease
from utils.eval import avg_inv_KLdiv, common_rows_proportion

def model_instantiate(trial: optuna.Trial, model_class: str, package: str, cv_folds: int, n_obs: int, seed: int):
    """
    Wrapper function: define class of generative model
    """
    
    # Define class of generative model 
    if model_class=='CTGAN':
        model = ctgan_instantiate(trial, package, cv_folds, n_obs, seed)
    elif model_class=='TVAE':
        model = tvae_instantiate(trial, package, cv_folds, n_obs, seed)
    else:
        raise ValueError(f'Invalid model class {model_class}')
    
    return model

def ctgan_instantiate(trial: optuna.Trial, package: str, cv_folds: int, n_obs: int, seed: int):
    """
    Wrapper function: define package used for CTGAN
    """
    
    # Batch size: fixed hyperparameter 
    training_folds_size = round((cv_folds-1)/cv_folds*n_obs) # round value to an integer
    batch_size = min(training_folds_size, 200)
    
    # Package to use
    if package=='tune_package':
        trial.suggest_categorical('package', ['synthcity', 'custom_sdv']) # if not predefined, then it is a (nested) hyperparameter as well
        trial.set_user_attr('package', trial.params['package'])
    elif package=='synthcity':
        trial.set_user_attr('package', 'synthcity') # fix package
    elif package=='custom_sdv':
        trial.set_user_attr('package', 'custom_sdv') # fix package
    
    # Define instantiation function
    if trial.user_attrs['package']=='synthcity':
        model = ctgan_synthcity_instantiate(trial, batch_size, seed)
    elif trial.user_attrs['package']=='custom_sdv':
        model = ctgan_custom_sdv_instantiate(trial, batch_size, seed)
    
    return model

def ctgan_synthcity_instantiate(trial: optuna.Trial, batch_size: int, seed: int) -> CTGANPlugin:
    """
    synthcity's CTGANPlugin: instantiate model with suggested hyperparameters
    """
    
    # Define hyperparameter space
    hyperparams = {
        # ARCHITECTURE
        'generator_n_layers_hidden': trial.suggest_int('generator_n_layers_hidden', low=1, high=4, step=1),
        'generator_n_units_hidden': trial.suggest_categorical('generator_n_units_hidden', [8, 16, 32, 64, 128, 256, 512]), 
        'discriminator_n_layers_hidden': trial.suggest_int('discriminator_n_layers_hidden', low=1, high=4, step=1),
        'discriminator_n_units_hidden': trial.suggest_categorical('discriminator_n_units_hidden', [8, 16, 32, 64, 128, 256, 512]), 
        # OPTIMIZATION
        'n_iter': trial.suggest_int('n_iter', low=5, high=300, log=True),
        'discriminator_n_iter': trial.suggest_categorical('discriminator_n_iter', [1, 5, 10]),
        'lr': trial.suggest_float('lr', low=1e-5, high=1e-2, log=True),
        'batch_size': batch_size,
        # REGULARIZATION
        'discriminator_dropout': trial.suggest_float('discriminator_dropout', low=0, high=1, log=False),
        # generator_dropout and weight_decay are not used downstream in CTGANPlugin so cannot be tuned
    }
    
    # n_iter_min is min number of iterations done before early stopping, so setting this to n_iter ensures that early stopping is not implemented
    hyperparams['n_iter_min'] = hyperparams['n_iter']
    
    # Define CTGAN model
    model = CTGANPlugin(**hyperparams, random_state=seed)
    
    return model

def ctgan_custom_sdv_instantiate(trial: optuna.Trial, batch_size: int, seed: int) -> CTGAN:
    """
    SDV's CTGAN (customized): instantiate model with suggested hyperparameters
    """
    
    hyperparams = {
        # ARCHITECTURE
        'generator_n_layers_hidden': trial.suggest_int('generator_n_layers_hidden', low=1, high=4, step=1),
        'generator_n_units_hidden': trial.suggest_categorical('generator_n_units_hidden', [8, 16, 32, 64, 128, 256, 512]),
        'discriminator_n_layers_hidden': trial.suggest_int('discriminator_n_layers_hidden', low=1, high=4, step=1),
        'discriminator_n_units_hidden': trial.suggest_categorical('discriminator_n_units_hidden', [8, 16, 32, 64, 128, 256, 512]),
        # OPTIMIZATION
        'epochs': trial.suggest_int('epochs', low=5, high=300, log=True),
        'discriminator_steps': trial.suggest_categorical('discriminator_steps', [1, 5, 10]),
        'generator_lr': trial.suggest_float('generator_lr', low=1e-5, high=1e-2, log=True),
        'batch_size': batch_size,
        # REGULARIZATION
        'generator_dropout': trial.suggest_float('generator_dropout', low=0, high=1, log=False),
        'discriminator_dropout': trial.suggest_float('discriminator_dropout', low=0, high=1, log=False),
        'generator_decay': trial.suggest_float('generator_decay', low=1e-6, high=1, log=True),
        'discriminator_decay': trial.suggest_float('discriminator_decay', low=1e-6, high=1, log=True),
    }
    
    # Set discriminator_lr equal to generator_lr
    hyperparams['discriminator_lr'] = hyperparams['generator_lr']
    trial.set_user_attr('discriminator_lr', hyperparams['discriminator_lr']) # fix discriminator_lr
    
    # Define generator and discriminator layer dimensions (tuple with dimensions of each layer is passed as a single argument)
    gen_dims = hyperparams['generator_n_layers_hidden']*(hyperparams['generator_n_units_hidden'],)
    discr_dims = hyperparams['discriminator_n_layers_hidden']*(hyperparams['discriminator_n_units_hidden'],)
    
    # Use the correct naming of hyperparameters in the module
    for name in ['discriminator_n_layers_hidden', 'discriminator_n_units_hidden', 'generator_n_layers_hidden', 'generator_n_units_hidden']:
        del hyperparams[name]
    hyperparams['generator_dim'] = gen_dims 
    hyperparams['discriminator_dim'] = discr_dims
    
    # Define CTGAN model
    model = CTGAN(**hyperparams, seed=seed)
    
    return model

def tvae_instantiate(trial: optuna.Trial, package: str, cv_folds: int, n_obs: int, seed: int):
    """
    Wrapper function: define package used for TVAE
    """
    
    # Batch size: fixed hyperparameter 
    training_folds_size = round((cv_folds-1)/cv_folds*n_obs) # round value to an integer
    batch_size = min(training_folds_size, 200)
    
    # Package to use
    if package=='tune_package':
        trial.suggest_categorical('package', ['synthcity', 'custom_sdv']) # if not predefined, then it is a (nested) hyperparameter as well
        trial.set_user_attr('package', trial.params['package'])
    elif package=='synthcity':
        trial.set_user_attr('package', 'synthcity') # fix package
    elif package=='custom_sdv':
        trial.set_user_attr('package', 'custom_sdv') # fix package
    
    # Define instantiation function
    if trial.user_attrs['package']=='synthcity':
        model = tvae_synthcity_instantiate(trial, batch_size, seed)
    elif trial.user_attrs['package']=='custom_sdv':
        model = tvae_custom_sdv_instantiate(trial, batch_size, seed)
    
    return model

def tvae_synthcity_instantiate(trial: optuna.Trial, batch_size: int, seed: int) -> TVAEPlugin:
    """
    synthcity's TVAEPlugin: instantiate model with suggested hyperparameters
    """
    
    # Define hyperparameter space
    hyperparams = {
        # ARCHITECTURE
        'n_units_embedding': trial.suggest_categorical('n_units_embedding', [32, 64, 128, 256, 512]),
        'n_layers': trial.suggest_int('n_layers', low=1, high=4, step=1),
        'n_hidden_units': trial.suggest_categorical('n_hidden_units', [32, 64, 128, 256, 512]),
        # OPTIMIZATION
        'n_iter': trial.suggest_int('n_iter', low=200, high=1000, log=True),
        # REGULARIZATION
        'loss_factor': trial.suggest_categorical('loss_factor', [1, 2, 5, 10]),
        'weight_decay': trial.suggest_float('weight_decay', low=1e-6, high=1, log=True), # shared between encoder and decoder 
        'encoder_dropout': trial.suggest_float('encoder_dropout', low=0, high=0.90, log=False), # generates NaN values if encoder_dropout is set too high
        'decoder_dropout': trial.suggest_float('decoder_dropout', low=0, high=0.90, log=False),
    }
    
    # Use the correct naming of hyperparameters in the module
    hyperparams['encoder_n_layers_hidden'] = hyperparams['n_layers']
    hyperparams['encoder_n_units_hidden'] = hyperparams['n_hidden_units']
    hyperparams['decoder_n_layers_hidden'] = hyperparams['n_layers'] # set decoder dimensions equal to encoder
    hyperparams['decoder_n_units_hidden'] = hyperparams['n_hidden_units'] # set decoder dimensions equal to encoder
    for name in ['n_layers', 'n_hidden_units']:
        del hyperparams[name]
        
    # n_iter_min is min number of iterations done before early stopping, so setting this to n_iter ensures that early stopping is not implemented
    hyperparams['n_iter_min'] = hyperparams['n_iter']
    
    # Define TVAE model
    model = TVAEPlugin(**hyperparams, random_state=seed)
    
    return model

def tvae_custom_sdv_instantiate(trial: optuna.Trial, batch_size: int, seed: int) -> TVAE:
    """
    SDV's TVAE (customized): instantiate model with suggested hyperparameters
    """
    
    hyperparams = {
        # ARCHITECTURE
        'embedding_dim': trial.suggest_categorical('embedding_dim', [32, 64, 128, 256, 512]),
        'n_layers': trial.suggest_int('n_layers', low=1, high=4, step=1), # combined with n_hidden_units to make compress_dims (encoder dimensions) and decompress_dims (decoder dimensions)
        'n_hidden_units': trial.suggest_categorical('n_hidden_units', [32, 64, 128, 256, 512]), # combined with n_layers to make compress_dims (encoder dimensions) and decompress_dims (decoder dimensions)
        # OPTIMIZATION
        'epochs': trial.suggest_int('epochs', low=200, high=1000, log=True),
        # REGULARIZATION
        'loss_factor': trial.suggest_categorical('loss_factor', [1, 2, 5, 10]),
        'weight_decay': trial.suggest_float('weight_decay', low=1e-6, high=1, log=True), # shared between encoder and decoder
        'encoder_dropout': trial.suggest_float('encoder_dropout', low=0, high=0.90, log=False), # generates NaN values if encoder_dropout is set too high
        'decoder_dropout': trial.suggest_float('decoder_dropout', low=0, high=0.90, log=False),
    }
    
    # Define encoder and decoder layer dimensions (here equal; tuple with dimensions of each layer is passed as a single argument)
    compress_dims = hyperparams['n_layers']*(hyperparams['n_hidden_units'],)
    decompress_dims = hyperparams['n_layers']*(hyperparams['n_hidden_units'],)
    
    # Use the correct naming of hyperparameters in the module
    for name in ['n_layers', 'n_hidden_units']:
        del hyperparams[name]
    hyperparams['compress_dims'] = compress_dims 
    hyperparams['decompress_dims'] = decompress_dims
    
    # Define TVAE model
    model = TVAE(**hyperparams, seed=seed)
    
    return model

def model_score(model: any=None, trial_package: str='synthcity', data: pd.DataFrame=None, 
                cv_folds: int=5, seed: int=0, discrete_columns: list[str]=None):
    """
    Cross-validated score using IKLD (+ sanity check)
    """
    
    # Implement k-fold cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    # Initialize an empty list to store the inverse KL divergence scores for each fold
    fidelity_syn, fidelity_gt, fidelity_bootstrap = [], [], []
    sanity_syn, sanity_gt, sanity_bootstrap = [], [], []
    
    for k, (train_idx, val_idx) in enumerate(kf.split(data)):
        
        # split in train and validation sets
        train_data = data.loc[train_idx, :] # training data (= in-fold original data)
        val_data = data.loc[val_idx, :] # validation data (= out-of-fold original data)
        
        # train generator and generate synthetic dataset
        if 'synthcity' in trial_package: # use synthcity plugins
            loader = GenericDataLoader(train_data, sensitive_features=list(train_data.columns))
            model.fit(loader)
            synthetic_data = model.generate(count=val_data.shape[0], random_state=seed).dataframe() # generated synthetic data = size of validation set (use same seed for model.fit as for model.generate)
        elif 'custom' in trial_package: # use custom plugins (adapted from SDV)
            model.fit(train_data, discrete_columns=discrete_columns) 
            synthetic_data = model.sample(n=val_data.shape[0]) # generated synthetic data = size of validation set (custom_sdv uses same seed for model.fit as for model.sample)
        
        # generate benchmarks (randomly sampled ground truth data and bootstrap of train data)
        gt_data = sample_disease(nsamples=val_data.shape[0], seed=k) # ground truth data sample = size of validation set
        bootstrap_data = train_data.sample(n=val_data.shape[0], replace=True, random_state=k) # bootstrap of (in-fold) (original) train data = size of validation set
        
        # fidelity metric: inverse KL divergence
        fidelity_syn.append(avg_inv_KLdiv(synthetic_data, val_data)) # between synthetic data and validation data
        fidelity_gt.append(avg_inv_KLdiv(gt_data, val_data)) # between ground truth data sample and validation data
        fidelity_bootstrap.append(avg_inv_KLdiv(bootstrap_data, val_data)) # between bootstrap of train data and validation data
        
        # sanity check: proportion of synthetic data records that are copied from real data records
        sanity_syn.append(common_rows_proportion(synthetic_data, train_data)) # between synthetic data and train data
        sanity_gt.append(common_rows_proportion(gt_data, train_data)) # between ground truth data sample and train data
        sanity_bootstrap.append(common_rows_proportion(bootstrap_data, train_data)) # between bootstrap of train data and train data
        
        # print metrics per fold
        #print(f'[Fold {k}] fidelity_syn: {fidelity_syn[-1]:.3f} (copy: {sanity_syn[-1]*100:.1f}%) - ' +
        #      f'fidelity_gt_sample: {fidelity_gt[-1]:.3f} (copy: {sanity_gt[-1]*100:.1f}%) - ' +
        #      f'fidelity_bootstrap: {fidelity_bootstrap[-1]:.3f} (copy: {sanity_bootstrap[-1]*100:.1f}%)')
    
    # Store and return scores per fold in dictionary
    cv_score = {'fidelity_syn': fidelity_syn, 'fidelity_gt': fidelity_gt, 'fidelity_bootstrap': fidelity_bootstrap,
                'sanity_syn': sanity_syn, 'sanity_gt': sanity_gt, 'sanity_bootstrap': sanity_bootstrap}
    
    return cv_score

def tuning_objective_single(trial: optuna.Trial, model_class: str=None, package: str='tune_package', data: pd.DataFrame=None,
                            cv_folds: int=5, seed: int=0, discrete_columns: list[str]=None, normalize_score: bool=False, print_params: bool=True) -> float:
    """
    Objective function for hyperparameter optimization over single seed
    """
    
    # Instantiate the generative model with the trial hyperparameters
    model = model_instantiate(trial, model_class, package, cv_folds, data.shape[0], seed) # seed is used for model initialization
    
    # Calculate score per fold using k-fold cross-validation
    cv_scores = model_score(model, trial.user_attrs['package'], data, cv_folds, seed, discrete_columns) 
    
    # Mean/std score over all folds
    score_syn_mean = np.mean(cv_scores['fidelity_syn']) # tuned model as generative model
    score_syn_std = np.std(cv_scores['fidelity_syn'])
    sanity_syn_mean = np.mean(cv_scores['sanity_syn'])
    
    score_gt_mean = np.mean(cv_scores['fidelity_gt']) # ground truth sample as generative model
    score_gt_std = np.std(cv_scores['fidelity_gt'])
    sanity_gt_mean = np.mean(cv_scores['sanity_gt'])
    
    score_bootstrap_mean = np.mean(cv_scores['fidelity_bootstrap']) # bootstrap sample as generative model
    score_bootstrap_std = np.std(cv_scores['fidelity_bootstrap'])
    sanity_bootstrap_mean = np.mean(cv_scores['sanity_bootstrap'])
    
    
    # print CV metrics
    print(f'[Trial {trial.number}, n={data.shape[0]}, seed={seed}] CV fidelity_syn: {score_syn_mean:.3f} (std: {score_syn_std:.3f}, copy: {sanity_syn_mean*100:.1f}%) - ' +
          f'CV fidelity_gt_sample: {score_gt_mean:.3f} (std: {score_gt_std:.3f}, copy: {sanity_gt_mean*100:.1f}%) - ' +
          f'CV fidelity_bootstrap: {score_bootstrap_mean:.3f} (std: {score_bootstrap_std:.3f}, copy: {sanity_bootstrap_mean*100:.1f}%)')
    
    # print trial hyperparameters
    if print_params:
        if(len(trial.params)>0):
            print('Hyperparameters: ' + str(trial.params)) # suggested trial hyperparameters
        else:
            print('Hyperparameters: ' + str(trial.system_attrs['fixed_params'])) # fixed trial hyperparameters
    
    # normalize score (with bootstrap sample as optimal generative benchmark)
    if normalize_score:
        score_syn_mean /= score_bootstrap_mean 
    
    return score_syn_mean # score_gt_mean (+- score_boostrap_mean if normalize_score==False) and sanity metrics are just informative and not used for hyperparameter selection

def tuning_objective_multiple(trial: optuna.Trial, model_class: str=None, package: str='tune_package', data: pd.DataFrame=None, 
                              cv_folds: int=5, seed: int=0, discrete_columns: list[str]=None, normalize_score: bool=True) -> float:
    """
    Objective function for hyperparameter optimization over multiple seeds
    """
    
    # initialize an empty list to store the k-fold cross-validated score for each seed
    scores = []
    normalize_text = 'normalized' if normalize_score else 'raw'
    
    # k-fold cross-validated score for each seed
    rng = np.random.default_rng(trial.number) # random number generator initialized by trial.number (for reproducibility) - overrules the seed argument
    seeds = rng.integers(1, 10000, size=5) # different set of seeds (used for model initialization and train/val split) for each trial
    for i in range(len(seeds)):
        score_syn = tuning_objective_single(trial, model_class, package, data, cv_folds, seeds[i], discrete_columns, normalize_score, print_params=False) # k-fold cross-validated score 
        scores.append(score_syn)
        trial.set_user_attr('score_seed' + str(i), score_syn) # store score per seed as user attribute (to retrieve later if needed) 
        # trial pruning
        score_ma = np.mean(scores) # moving average of scores
        trial.report(score_ma, i) 
        if trial.should_prune(): # handle pruning based on the moving average of scores
            print(f'--> [Trial {trial.number}] Trial pruned: {normalize_text} moving average of {score_ma:.3f} after ' + str(i+1) + ' seed initialization(s)')
            raise optuna.TrialPruned()
     
    # mean/std score over all seeds
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    trial.set_user_attr('score_mean', score_mean) # store as user attribute (to retrieve later if needed)
    trial.set_user_attr('score_std', score_std) # store as user attribute (to retrieve later if needed)
    
    # print metrics
    print(f'--> [Trial {trial.number}, summary] {normalize_text} CV fidelity_syn: {scores} (mean: {score_mean:.3f}, std: {score_std:.3f})')
    
    # print trial hyperparameters
    if(len(trial.params)>0):
        print('Hyperparameters: ' + str(trial.params)) # suggested trial hyperparameters
    else:
        print('Hyperparameters: ' + str(trial.system_attrs['fixed_params'])) # fixed trial hyperparameters
    
    return score_mean