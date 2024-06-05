import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.stats import entropy
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.api import OLS
import warnings
import itertools
from itertools import product
import plotnine
from plotnine import *
from mizani.formatters import percent_format 
from mizani.transforms import trans

def dataframe2probs(X_orig: pd.DataFrame, X_synth: pd.DataFrame, max_bins: int=10) -> dict:
    """
    Get percentual frequencies for each possible real categorical value.
    Continuous variables are binned with min(max_bins, #unique values) bins.
    Based on get_frequency from https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/_utils.py
    """
    res = {}
    for col in X_orig.columns:
        local_bins = min(max_bins, len(X_orig[col].unique()))

        # categorical? only conditions on number 5 in synthcity; add type checking
        if len(X_orig[col].unique()) < 6 or not np.issubdtype(X_orig[col].dtype, np.number):
            orig = (X_orig[col].value_counts() / len(X_orig)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            orig_vals, bins = np.histogram(X_orig[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins) #same number of bins as ground truth bins
            orig = {k: v / (sum(orig_vals) + 1e-8) for k, v in zip(bins, orig_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in orig:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in orig or orig[val] == 0:
                orig[val] = 1e-11

        if orig.keys() != synth.keys():
            raise ValueError(f'Invalid features. {orig.keys()}. syn = {synth.keys()}')
        res[col] = (list(orig.values()), list(synth.values()))

    return res

def all_KLdivs(X_orig: pd.DataFrame, X_synth: pd.DataFrame, max_bins: int=10) -> dict:
    """
    Calculate the Shannon entropy/relative entropy.
    """
    probs = dataframe2probs(X_orig, X_synth, max_bins=max_bins)
    res = dict()
    for col in X_orig.columns:
        p_orig, p_synth = probs[col]
        res[col] = entropy(p_orig, qk=p_synth)
    return res

def avg_inv_KLdiv(X_orig: pd.DataFrame, X_synth: pd.DataFrame, max_bins: int=10) -> float:
    """
    average of smooth inverse KL-divergence: avg_over_variables(1 / (1 + KLdiv_variable(orig||synthetic)))
    """
    smooth_inverse_KL_divs = 1 / (1 + np.array(list(all_KLdivs(X_orig, X_synth, max_bins=max_bins).values())))
    return np.mean(smooth_inverse_KL_divs)

def common_rows_proportion(X_orig: pd.DataFrame, X_synth: pd.DataFrame) -> float:
    """
    Based on CommonRowsProportion from https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_sanity.py

    Returns the proportion of rows in the synthetic dataset that are copies of the original dataset.
    Score:
        0: there are no common rows between the real and synthetic datasets.
        1: all rows of the synthetic dataset are copies of the original dataset.
    """
    if len(X_orig.columns) != len(X_synth.columns):
        raise ValueError(f'Incompatible dataframe {X_orig.shape} and {X_synth.shape}')

    intersection = (
        X_orig.merge(X_synth, how="inner", indicator=False) # do not use .drop_duplicates() as in original synthcity code 
    )
        
    return len(intersection) / len(X_orig)

def estimate_estimator(data, var, estimator):
    """
    Calculate estimate for each dataset.
    """    
    
    datasets = [data[name] for name in data if name not in ['meta_data']]

    if estimator == 'mean':
        return [np.mean(dataset[var]) for dataset in datasets]
    elif estimator == 'mean_se':
        return [np.std(dataset[var], ddof=1)/np.sqrt(len(dataset[var])) for dataset in datasets]

    elif estimator == 'sd':
        return [np.std(dataset[var], ddof=1) for dataset in datasets]

    elif estimator == 'var':
        return [np.var(dataset[var], ddof=1) for dataset in datasets]

    elif estimator == 'prop':
        return [np.mean(dataset[var].astype(int)) for dataset in datasets]
    elif estimator == 'prop_se':
        return [np.sqrt((np.mean(dataset[var].astype(int))*(1-np.mean(dataset[var].astype(int))))/len(dataset[var])) for dataset in datasets]

    if estimator == 'polr':
        output = []
        for dataset in datasets:
            try:
                model = OrderedModel(dataset[var[0]], dataset.loc[:,var[1:]].astype(float), distr='logit').fit(method='bfgs', disp=False) # var[0] is outcome, var[1:] are predictors
                indices = model.cov_params().index.get_indexer(var[1:])
                output.append(np.hstack((-model.params.to_numpy()[indices], # coefficients
                                         np.sqrt(model.cov_params().to_numpy().diagonal()[indices])))) # SEs
            except Exception as e:
                output.append(np.repeat(np.nan, len(var[1:])*2))
        return output

    elif estimator == 'gamr':
        output = []
        for dataset in datasets:
            try:
                with warnings.catch_warnings(): 
                    # ignore warning on line 307 of statsmodels/genmod/generalized_linear_model.py: 
                    # DomainWarning: The InversePower link function does not respect the domain of the Gamma family.
                    warnings.filterwarnings('ignore', lineno=307)
                    model = GLM(dataset[var[0]], sm.add_constant(dataset.loc[:,var[1:]].astype(float)), 
                                family=sm.families.Gamma(sm.families.links.InversePower())).fit() # var[0] is outcome, var[1:] are predictors
                indices = model.cov_params().index.get_indexer(var[1:])
                output.append(np.hstack((model.params.to_numpy()[indices], # coefficients
                                         np.sqrt(model.cov_params().to_numpy().diagonal()[indices])))) # SEs
            except Exception as e:
                output.append(np.repeat(np.nan, len(var[1:])*2))
        return output

    elif estimator == 'logr':
        output = []
        for dataset in datasets:
            try:
                model = GLM(dataset[var[0]], sm.add_constant(dataset.loc[:,var[1:]].astype(float)), 
                            family=sm.families.Binomial(link=sm.families.links.Logit())).fit() # var[0] is outcome, var[1:] are predictors
                indices = model.cov_params().index.get_indexer(var[1:])
                output.append(np.hstack((model.params.to_numpy()[indices], # coefficients
                                         np.sqrt(model.cov_params().to_numpy().diagonal()[indices])))) # SEs
            except Exception as e:
                output.append(np.repeat(np.nan, len(var[1:])*2))
        return output

    else:
        return [np.nan for dataset in datasets]

def CI_coverage(estimate, se, ground_truth, se_correct_factor=1, distribution='standardnormal', df=1, quantile=0.975):
    """
    Calculate coverage of a confidence interval based on t- or standard normal distribution.
    """
    
    # Define quantile based on distribution
    if distribution == 't':
        q = ss.t.ppf(quantile, df=df)
    elif distribution == 'standardnormal': 
        q = ss.norm.ppf(quantile)
    else:
        raise ValueError('Choose \'t\' or \'standardnormal\' distribution')
    
    # Check if confidence interval contains ground_truth
    if (estimate-q*se_correct_factor*se <= ground_truth) & (ground_truth <= estimate+q*se_correct_factor*se):
        coverage = True
    else:
        coverage = False
    
    return coverage

def q1(x):
    """
    First quartile.
    """ 
    return x.quantile(0.25)

def q3(x):
    """
    Third quartile.
    """ 
    return x.quantile(0.75)

def expand_grid(data_dict):
    """
    Expand grid.
    """ 
    rows = product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def missing_se(meta_data: pd.DataFrame=None, select_se: list=[]):
    """
    Extract missing SEs.
    """ 
    
    # Select SEs
    select_se = [column for column in meta_data.columns if column[-3:]=='_se'] # select columns with '_se' suffix
    
    # Missing SEs
    output_data = meta_data.melt(id_vars=['n', 'run', 'generator'],
                                 value_vars=select_se,
                                 var_name='estimator',
                                 value_name='value').query('value.isna()').drop(columns=['value'])
    
    return output_data

def plot_bias(meta_data: pd.DataFrame, select_estimators: list=[], plot_outliers: bool=True, order_generators: list=[], figure_size: tuple=(), 
              unit_rescale: dict={}, plot_estimates: bool=False, ground_truth: dict={}):
    """
    Plot bias.
    If plot_estimates==True, then ground_truth should be given.
    """ 
    
    # Select estimators
    suffix = '_bias'
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if suffix in column] # select all columns with '_bias' suffix
    else:
        select_estimators = [estimator + suffix for estimator in select_estimators] # add suffix '_bias'
    
    # Plot estimates instead of bias
    if plot_estimates:
        select_estimators = [estimator[:-len('_bias')] for estimator in select_estimators] # remove suffix '_bias'
        suffix = ''
    
    # Average bias/estimate of estimator for population parameter per generator (over sets) for each n and run
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias')
    
    # Don't plot outliers based on IQR of empirical SE (set plot_outliers = False)
    if plot_outliers:
        bias_data['plot'] = True
    else:
        iqr = meta_data.groupby(['n', 'generator'])[select_estimators].agg([q1, q3]).reset_index()
        iqr.columns = ['_'.join(column) if column[1] != '' else column[0] for column in iqr.columns.to_flat_index()] # rename hierarchical columns
        iqr = iqr.melt(
            id_vars=['n', 'generator'],
            var_name='estimator',
            value_name='value')
        iqr['quantile'] = iqr.apply(lambda x: 'q1' if 'q1' in x['estimator'] else 'q3', axis=1)
        iqr['estimator'] = list(map(lambda x: x[:-3], iqr['estimator']))
        iqr = iqr.pivot(index=['n', 'generator', 'estimator'], columns='quantile', values='value').reset_index()
        bias_data = bias_data.merge(iqr, how='left', on=['n', 'generator', 'estimator'])
        bias_data['plot'] = bias_data.apply(
            lambda x: x['bias'] > x['q1'] - 1.5*(x['q3']-x['q1']) and x['bias'] < x['q3'] + 1.5*(x['q3']-x['q1']), axis=1) # non-outlier is > Q1-1.5*IQR and < Q3+1.5*IQR
    
    # Change plotting order (non-alphabetically) of estimator and generator
    bias_data['estimator'] = pd.Categorical(list(map(lambda x: x[:(-len(suffix) or None)], bias_data['estimator'])), # remove suffix; 'or None' is called when suffix=''
                                            categories=[estimator[:(-len(suffix) or None)] for estimator in list(bias_data['estimator'].unique())]) # change order (non-alphabetically); 'or None' is called when suffix=''
    if len(order_generators)!=0:
        bias_data['generator'] = pd.Categorical(bias_data['generator'], 
                                                categories=order_generators) # change order (non-alphabetically)
    
    # Root-n consistency funnel
    root_n_consistency = expand_grid({'x': np.arange(np.min(bias_data['n']), np.max(bias_data['n'])), 'estimator': bias_data['estimator'].unique()})
    root_n_consistency['estimator'] = pd.Categorical(root_n_consistency['estimator'], categories=root_n_consistency['estimator'].unique()) # change order (non-alphabetically)
    root_n_consistency['unit_sd'] = root_n_consistency.apply(lambda x: unit_rescale[x['estimator']], axis=1) # asymptotic variance
    root_n_consistency['y'] = root_n_consistency.apply(lambda x: ground_truth[x['estimator']], axis=1) if plot_estimates else 0 
    root_n_consistency['y_ul'] = root_n_consistency['y'] + ss.norm.ppf(0.975)*root_n_consistency['unit_sd']/np.sqrt(root_n_consistency['x'])
    root_n_consistency['y_ll'] = root_n_consistency['y'] - ss.norm.ppf(0.975)*root_n_consistency['unit_sd']/np.sqrt(root_n_consistency['x'])
    
    # Labs
    plot_title = 'Estimates of estimator' if plot_estimates else 'Bias of estimator'
    plot_y_lab = 'Estimate' if plot_estimates else 'Bias'
    
    # Default figure size
    if len(figure_size) == 0:
        figure_size = (1.5+len(bias_data['generator'].unique())*1.625, 1.5+len(bias_data['estimator'].unique())*1.625)
    
    # Plot average bias/estimate and root-n consistency funnel
    plot = ggplot(bias_data.query('plot==True'), aes(x='n', y='bias', colour='generator')) +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y'), linetype='dashed', colour='black') +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y_ul'), linetype='dashed', colour='black') +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y_ll'), linetype='dashed', colour='black') +\
        geom_point(alpha=0.20) +\
        stat_summary(geom='line') +\
        scale_x_continuous(breaks=list(bias_data['n'].unique()), labels=list(bias_data['n'].unique()), trans='log') +\
        facet_grid('estimator ~ generator', scales='free') +\
        scale_colour_manual(values={'original': '#808080', 'synthpop': '#1E64C8', 'bayesian_network': '#825491', 'bayesian_network_DAG': '#71A860',
                                    'ctgan': '#DC4E28', 'custom_ctgan': '#F1A42B', 'tvae': '#8BBEE8', 'custom_tvae': '#FFD200'}) +\
        labs(title=plot_title, x='n (log scale)', y=plot_y_lab) +\
        theme_bw() +\
        theme(aspect_ratio=1,
              plot_title=element_text(hjust=0.5, size=12), # title size
              axis_title=element_text(size=10), # axis title size
              strip_text=element_text(size=8), # facet_grid title size
              axis_text=element_text(size=8), # axis labels size
              legend_position='none',
              figure_size=figure_size)
    
    return plot

class log_squared(trans):
    """
    Custom axis transformation: transform x to log(x**2).
    """
    @staticmethod
    def transform(x):
        return np.log(x**2)
    
    @staticmethod
    def inverse(x):
        return np.sqrt(np.exp(x))

def plot_convergence_rate(meta_data: pd.DataFrame, select_estimators: list=[], figure_ncol: int=None, order_generators: list=[], figure_size: tuple=(), 
                          unit_rescale: dict={}, metric: str='se', check_root_n: bool=True):
    """
    Plot convergence rate.
    """ 
    
    # Select estimators
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if '_bias' in column] # select all columns with '_bias' suffix
    else:
        select_estimators = [estimator + '_bias' for estimator in select_estimators] # add suffix '_bias'
                          
    # Empirical SE/bias of estimator for population parameter per generator (over sets) for each n and run
    if metric == 'se':
        plot_data = meta_data.groupby(['n', 'generator'])[select_estimators].apply(lambda x: np.std(x, ddof=1)).reset_index().melt(
            id_vars=['n', 'generator'],
            var_name='estimator',
            value_name='metric')
        if check_root_n:
            plot_y_axis = 'se_rescaled*np.sqrt(n)'
            plot_y_lab = 'empirical SE * sqrt(n)'
            plot_x_axis = 'n' # the x-axis will be transformed to log-scale in the plot
            plot_x_lab = 'n (log scale)'
            plot_x_trans = 'log'
        else:
            plot_y_axis = 'np.log(metric)'
            plot_y_lab = 'log(empirical SE)'
            plot_x_axis = 'n' # the x-axis will be transformed to log-scale in the plot
            plot_x_lab = 'n (log scale)'
            plot_x_trans = 'log'
            
    elif metric == 'bias':
        plot_data = meta_data.groupby(['n', 'generator'])[select_estimators].mean().reset_index().melt(
            id_vars = ['n', 'generator'],
            var_name = 'estimator',
            value_name = 'metric')
        if check_root_n:
            plot_y_axis = 'se_rescaled*n'
            plot_y_lab = 'empirical SE * n'
            plot_x_axis = 'n' # the x-axis will be transformed to log-scale in the plot
            plot_x_lab = 'n (log scale)'
            plot_x_trans = 'log'
        else:
            plot_y_axis = 'np.log(metric**2)'
            plot_y_lab = 'log(empirical bias**2)'
            plot_x_axis = 'n**2' # the x-axis will be transformed to log-scale in the plot
            plot_x_lab = 'n**2 (log scale)'
            plot_x_trans = log_squared
    
    # Rescale estimate with asymptotic variance
    plot_data['unit_sd'] = plot_data.apply(lambda x: unit_rescale[x['estimator'][:-len('_bias')]], axis=1)
    plot_data['se_rescaled'] = plot_data['metric']/plot_data['unit_sd']
    
    # Custom breaks and labels
    plot_x_breaks = list(plot_data['n'].unique()**2) if (metric=='bias' and not check_root_n) else list(plot_data['n'].unique())
    plot_x_labels = list(plot_data['n'].unique())
    
    # Change plotting order (non-alphabetically) of estimator and generator
    plot_data['estimator'] = pd.Categorical(plot_data['estimator'], categories=list(plot_data['estimator'].unique())) # change order (non-alphabetically)
    if len(order_generators)!=0:
        plot_data['generator'] = pd.Categorical(plot_data['generator'], categories=order_generators) # change order (non-alphabetically)
    
    # Default figure columns and size
    if figure_ncol == None:
        figure_ncol = 7
        
    if len(figure_size) == 0:
        figure_size = (16,8)
    
    # Plot
    plot = ggplot(plot_data, aes(x=plot_x_axis, y=plot_y_axis, colour='generator'))
    
    # Plot root-n-convergence of naive and corrected SE (if metric == 'se' and not check_root_n)
    if metric == 'se' and not check_root_n:
        
        # Create additional dataframe with naive and corrected SE
        constant_c = plot_data.drop_duplicates(subset=['estimator', 'unit_sd'])[['estimator', 'unit_sd']]
        constant_c['formula_se'] = 'naive'
        constant_c_corrected = constant_c.copy()
        constant_c_corrected['formula_se'] = 'corrected'
        constant_c_corrected['unit_sd'] *= np.sqrt(2)
        constant_c = pd.concat([constant_c, constant_c_corrected], axis=0)
        constant_c['formula_se'] = pd.Categorical(constant_c['formula_se'], categories=list(constant_c['formula_se'].unique()))
        
        # Add abline to plot
        plot += geom_abline(data=constant_c, mapping=aes(intercept='np.log(unit_sd)', slope=-0.5, linetype='formula_se'), colour='black') 
    
    # Continue plot building
    plot = plot +\
        geom_point() +\
        stat_smooth(method='lm', se=False) +\
        scale_x_continuous(breaks=plot_x_breaks, labels=plot_x_labels, trans=plot_x_trans) +\
        facet_wrap('estimator', ncol=figure_ncol, scales='free') +\
        scale_colour_manual(values={'original': '#808080', 'synthpop': '#1E64C8', 'bayesian_network': '#825491', 'bayesian_network_DAG': '#71A860',
                                    'ctgan': '#DC4E28', 'custom_ctgan': '#F1A42B', 'tvae': '#8BBEE8', 'custom_tvae': '#FFD200'}) +\
        labs(title='Convergence of estimator', x=plot_x_lab, y=plot_y_lab) +\
        theme_bw() +\
        theme(aspect_ratio=2/3,
              plot_title=element_text(hjust=0.5, size=12), # title size
              axis_title=element_text(size=10), # axis title size
              strip_text=element_text(size=8), # facet_grid title size
              axis_text=element_text(size=8), # axis labels size
              figure_size=figure_size)
    
    # Add legend for root-n-convergence of naive and corrected SE (if metric == 'se' and not check_root_n)
    if metric == 'se' and not check_root_n:
        plot += scale_linetype_manual(values={'naive': 'dashed', 'corrected': 'dotted'})
    
    return plot


def calculate_conv_rate(data: pd.DataFrame, generator: str=None, estimator: str=None,
                        intercept: bool=True, unit_rescale: dict={}, metric: str='se', round_decimals: int=2, show_ci: bool=False, quantile: int=0.975):
    """
    Calculate convergence rate a in c/(n**a).
    """         
    # SE: log[sd(estimate)] = log(c) - a*log(n), or equivalently log[sd(estimate-groundtruth)] = log(c) - a*log(n)
    if metric == 'se':
        empirical_se = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].apply(lambda x: np.std(x, ddof=1)).reset_index()
        if intercept:
            log_lr = OLS(np.log(empirical_se['bias']), sm.add_constant(np.log(empirical_se['n']))).fit() # add intercept
        else:
            unit_sd = unit_rescale[estimator] # rescale estimate with asymptotic variance
            log_lr = OLS(np.log(empirical_se['bias']/unit_sd), np.log(empirical_se['n'])).fit() # an intercept is not included by default
        
    # Bias: log[mean(estimate-groundtruth)**2] = log(c) - a*log(n**2)
    elif metric == 'bias':
        empirical_bias = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].mean().reset_index()
        if intercept:
            log_lr = OLS(np.log(empirical_bias['bias']**2), sm.add_constant(np.log(empirical_bias['n']**2))).fit() # add intercept
        else:
            unit_sd = unit_rescale[estimator] # rescale estimate with asymptotic variance
            log_lr = OLS(np.log((empirical_bias['bias']/unit_sd)**2), np.log(empirical_bias['n']**2)).fit() # an intercept is not included by default
           
    # Extract rate and rate_se
    coeff_index = 1 if intercept else 0
    rate = -log_lr.params[coeff_index]
    rate_se = np.sqrt(log_lr.cov_params().to_numpy().diagonal()[coeff_index])
    rate_q = ss.t.ppf(quantile, df=(log_lr.nobs-1))
    rate_ll = rate-rate_q*rate_se
    rate_ul = rate+rate_q*rate_se
        
    # Round decimals
    rate = f'{rate:.{round_decimals}f}'
    rate_ll = f'{rate_ll:.{round_decimals}f}'
    rate_ul = f'{rate_ul:.{round_decimals}f}'
        
    output = f'{rate} [{rate_ll}; {rate_ul}]' if show_ci else rate
        
    return output

def table_convergence_rate(meta_data: pd.DataFrame, select_estimators: list=[], 
                           intercept: bool=False, unit_rescale: dict={}, metric: str='se', round_decimals: int=2, show_ci: bool=False, quantile: int=0.975):
    """
    Table with convergence rates.
    """ 
    
    # Select estimators
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if '_bias' in column] # select all columns with '_bias' suffix
    else:
        select_estimators = [estimator + '_bias' for estimator in select_estimators] # add suffix '_bias'
    
    # Average bias of estimator for population parameter per generator (over sets) for each n and run
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias')
    bias_data['estimator'] = list(map(lambda x: x[:-len('_bias')], bias_data['estimator'])) # remove '_bias' suffix
    
    # Calculate convergence rate for every estimator
    output_data = expand_grid({'generator': bias_data['generator'].unique(),
                               'estimator': bias_data['estimator'].unique()})
    output_data['convergence rate'] = output_data.apply(
        lambda x: calculate_conv_rate(data=bias_data, generator=x['generator'], estimator=x['estimator'], 
                                      intercept=intercept, unit_rescale=unit_rescale, metric=metric, round_decimals=round_decimals, show_ci=show_ci, quantile=quantile), axis=1)
    
    return output_data

def se_underestimation(meta_data: pd.DataFrame, select_estimators: list=[], correction_factor: int=1):
    """
    Calculate underestimation of empirical SE by model-based SE.
    """
    
    # Select estimators
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if 'bias' in column] # select all columns with '_bias' suffix
    else:
        select_estimators = [estimator + '_bias' for estimator in select_estimators] # add suffix '_bias'
    
    # Average bias of estimator for population parameter per generator (over sets) for each n and run
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias') 
    
    # Create dataset
    output_data = expand_grid({'generator': bias_data['generator'].unique(),
                               'estimator': bias_data['estimator'].unique()})
    
    # Empirical SE
    empirical_se = output_data.apply(lambda x: 
                                     bias_data.query('generator==\'' + x['generator'] + '\' & estimator==\'' + x['estimator'] + '\'').groupby('n')['bias'].apply(lambda x: np.std(x, ddof=1)),
                                     axis=1) 
    
    # Model-based SE
    model_based_se = output_data.apply(lambda x: 
                                       meta_data.query('generator==\'' + x['generator'] + '\'').groupby('n')[x['estimator'].replace('_bias', '_se')].mean() * correction_factor,
                                       axis=1) 
    
    # Metric 
    metric = (model_based_se-empirical_se)/empirical_se
    
    # Add metric to dataset
    output_data = pd.concat([output_data, metric], axis=1)
    
    return output_data

def summary_table(meta_data: pd.DataFrame, select_estimators: list=[], ground_truth: dict={}, correction_factor: int=1):
    """
    Create summary table.
    """ 
    
    # Select estimators
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if 'bias' in column] # select all columns with '_bias' suffix
        select_estimators_no_suffix = [name[:-len('_bias')] for name in select_estimators] # remove '_bias' suffix
    else:
        select_estimators_no_suffix = select_estimators
        select_estimators = [estimator + '_bias' for estimator in select_estimators] # add suffix '_bias'
        
    # Average estimate of population parameter per generator (over sets and runs) for each n
    estimate_data = meta_data.groupby(['n', 'generator'])[select_estimators_no_suffix].mean().reset_index().melt(
        id_vars=['n', 'generator'],
        var_name='estimator',
        value_name='average')
    
    # Average absolute bias of estimator for population parameter per generator (over sets and runs) for each n
    bias_data = meta_data.groupby(['n', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'generator'],
        var_name='estimator',
        value_name='absolute_bias')
    bias_data['estimator'] = list(map(lambda x: x[:-len('_bias')], bias_data['estimator'])) # remove '_bias' suffix
    
    # Average relative bias
    bias_data['relative_bias'] = bias_data.apply(lambda x: x['absolute_bias']/ground_truth[x['estimator']], axis=1) 
    
    # Merge estimate_data and bias_data
    output_data = estimate_data.merge(bias_data, how='left', on=['n', 'generator', 'estimator'])
    
    # SE underestimation
    se_underest = se_underestimation(meta_data, select_estimators_no_suffix, correction_factor=correction_factor).melt(
        id_vars=['estimator', 'generator'],
        var_name='n',
        value_name='SE_underestimation')
    se_underest['estimator'] = list(map(lambda x: x[:-len('_bias')], se_underest['estimator'])) # remove '_bias' suffix
    
    # Merge output_data and se_underest
    output_data = output_data.merge(se_underest, how='left', on=['n', 'generator', 'estimator'])
    
    return output_data

def plot_type_I_II_error(meta_data: pd.DataFrame, select_estimator: str, order_generators: list=[], use_power: bool=False, plot_intercept: bool=True, figure_size: tuple=()):
    """
    Plot type I and II error rate.
    """
    
    # Select estimators
    select_estimators = [column for column in meta_data.columns if select_estimator + '_NHST' in column] # add suffix '_NHST'
    
    # Average type 1 and type 2 error rate per generator (over sets and runs) for each n
    NHST_data_plot = meta_data.groupby(['n', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'generator'],
        var_name='error',
        value_name='probability')
    
    # Change plotting order (non-alphabetically)
    NHST_data_plot['corrected'] = pd.Categorical(list(map(lambda x: 'corrected SE' if 'corrected' in x else 'model-based SE', NHST_data_plot['error'])), # change label
                                                 categories=['model-based SE', 'corrected SE']) # change order (non-alphabetically)
    if len(order_generators)!=0:
        NHST_data_plot['generator'] = pd.Categorical(NHST_data_plot['generator'],
                                                     categories=order_generators) # change order (non-alphabetically)
    
    # Additional plot formatting 
    NHST_data_plot['error'] = pd.Categorical(list(map(lambda x: 'type 1 error' if 'type1' in x else ('power' if use_power else 'type 2 error'), NHST_data_plot['error'])), # change label
                                             categories=['type 1 error', 'power' if use_power else 'type 2 error']) # change order (non-alphabetically) 
    NHST_data_plot['probability'] = NHST_data_plot.apply(lambda x: 1-x['probability'] if x['error']=='power' else x['probability'], axis=1) # power = 1 - type 2 error
    
    # Default figure size
    if len(figure_size) == 0:
        figure_size = (8,6)
        
    # Plot
    plot = ggplot(NHST_data_plot, aes(x='n', y='probability', colour='generator', linetype='generator'))
    
    # Plot intercepts (if plot_intercept)
    if plot_intercept:
        intercepts = pd.DataFrame({'error': ['type 1 error', 'power' if use_power else 'type 2 error'], 'intercept': [0.05, 0.80 if use_power else 0.20]}) # add horizontal lines
        intercepts['error'] = pd.Categorical(intercepts['error'], categories=['type 1 error', 'power' if use_power else 'type 2 error']) # change order (non-alphabetically) 
        plot += geom_hline(data=intercepts, mapping=aes(yintercept='intercept'), linetype='dashed')
    
    # Continue plot building
    plot = plot +\
        geom_line() +\
        scale_x_continuous(breaks=list(NHST_data_plot['n'].unique()), labels=list(NHST_data_plot['n'].unique()), trans='log') +\
        scale_y_continuous(limits=(0,1), labels=percent_format()) +\
        facet_grid('error ~ corrected') +\
        scale_colour_manual(values={'original': '#808080', 'synthpop': '#1E64C8', 'bayesian_network': '#825491', 'bayesian_network_DAG': '#71A860',
                                    'ctgan': '#DC4E28', 'custom_ctgan': '#F1A42B', 'tvae': '#8BBEE8', 'custom_tvae': '#FFD200'}) +\
        scale_linetype_manual(values={'original': 'solid', 'synthpop': 'solid', 'bayesian_network': 'solid', 'bayesian_network_DAG': 'solid',
                                      'ctgan': 'dashed', 'custom_ctgan': 'dashed', 'tvae': 'dashed', 'custom_tvae': 'dashed'}) +\
        labs(title='Type 1 error and ' + ('power' if use_power else 'type 2 error') + ' for ' + select_estimator, x='n (log scale)') +\
        theme_bw() +\
        theme(plot_title=element_text(hjust=0.5, size=12), # title size
              axis_title=element_text(size=10), # axis title size
              strip_text=element_text(size=8), # facet_grid title size
              axis_text=element_text(size=8), # axis labels size
              figure_size=figure_size)
    
    return plot

def coverage_run_plot(meta_data: pd.DataFrame, select_estimator: str, order_generators: list=[], ground_truth: dict={}, 
                      quantile: int=0.975, n_runs: int=10, figure_size: tuple=()):
    """
    Plot confidence interval coverage of the first Monte Carlo runs.
    """
    
    # Reformating 
    plot_data = meta_data.head(n=n_runs*3).copy() # select first n_runs
    plot_data['generator'] = pd.Categorical(plot_data['generator'], categories=order_generators) # reorder categories
    plot_data['run'] = pd.Categorical(list(map(lambda x: str(int(x[len('run_'):])+1), plot_data['run'])), # change label 
                                      categories=[str(i+1) for i in range(n_runs)]) # reorder categories
    
    # Add correction factor
    plot_data = pd.concat([plot_data.assign(correction_factor = 1, formula_SE='Model-based SE'),
                           plot_data.assign(correction_factor = np.sqrt(2), formula_SE='Corrected SE')], axis=0)
    plot_data['formula_SE'] = pd.Categorical(plot_data['formula_SE'], categories=['Model-based SE', 'Corrected SE']) # reorder categories
    
    # Calculate confidence intervals
    plot_data[select_estimator + '_q'] = plot_data.apply(lambda i: ss.t.ppf(quantile, df=i['n']-1) if 'mean' in select_estimator else ss.norm.ppf(quantile), axis=1) # calculate quantile for CI
    plot_data[select_estimator + '_ll'] = plot_data.apply(lambda i: i[select_estimator] - i['correction_factor']*i[select_estimator + '_q']*i[select_estimator + '_se'], axis=1) # lower limit of CI
    plot_data[select_estimator + '_ul'] = plot_data.apply(lambda i: i[select_estimator] + i['correction_factor']*i[select_estimator + '_q']*i[select_estimator + '_se'], axis=1) # upper limit of CI
    
    # Add ground truth to each facet
    intercepts = pd.DataFrame({'generator': plot_data['generator'].unique(),
                               'estimator': [select_estimator]*len(plot_data['generator'].unique()),
                               'intercept': [ground_truth[select_estimator]]*len(plot_data['generator'].unique())})
    
    # Default figure size
    if len(figure_size) == 0:
        figure_size = (8,6)
    
    # Plot
    plot = ggplot(plot_data.query('not (generator==\'original\' & formula_SE==\'Corrected SE\')'), 
                                 aes(y=select_estimator, x='run', colour='generator', group='formula_SE', alpha='formula_SE')) +\
    geom_hline(data=intercepts, mapping=aes(yintercept='intercept', linetype='generator'), colour='black') +\
    geom_point(position=position_dodge(0.75)) +\
    scale_colour_manual(values={'original': '#808080', 'synthpop': '#1E64C8', 'bayesian_network': '#825491', 'bayesian_network_DAG': '#71A860',
                                'ctgan': '#DC4E28', 'custom_ctgan': '#F1A42B', 'tvae': '#8BBEE8', 'custom_tvae': '#FFD200'}) +\
    scale_linetype_manual(values={'original': 'dashed', 'synthpop': 'dashed', 'bayesian_network': 'dashed', 'bayesian_network_DAG': 'dashed',
                                  'ctgan': 'dashed', 'custom_ctgan': 'dashed', 'tvae': 'dashed', 'custom_tvae': 'dashed'}, breaks=['original'], labels=['Population parameter']) +\
    scale_alpha_manual(values={'Model-based SE': 1, 'Corrected SE': 0.5}) +\
    geom_errorbar(aes(ymin=select_estimator + '_ll', ymax=select_estimator + '_ul', group='formula_SE'), position = position_dodge(0.75)) +\
    facet_wrap('generator') +\
    labs(x='Run') +\
    guides(alpha=guide_legend(title='', reverse=True), 
           colour=False,
           linetype=guide_legend(title='')) +\
    theme_bw() +\
    theme(plot_title=element_blank(), # title size
          axis_title=element_text(size=12), # axis title size
          strip_text=element_text(size=10), # facet_grid title size
          axis_text=element_text(size=10), # axis labels size
          legend_position='bottom',
          legend_title=element_blank(), # legend title size
          legend_text=element_text(size=10), # legend labels size
          figure_size=figure_size) +\
    coord_flip()
    
    return plot