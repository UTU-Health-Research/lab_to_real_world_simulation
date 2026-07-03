import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./article.mplstyle')
import model_functions
from monte_carlo_model import run_monte_carlo
from utils import load_yaml
from scipy import stats
from plotting import plotting


#loading arguments from yaml files
args_lab = load_yaml('experiments/experiment_lab.yaml')
args_real = load_yaml('experiments/experiment_real.yaml')

#generating prevalence vector
prevalence = model_functions.generate_prevalence()

def monte_carlo(number_of_simulations: int = 5):
    """ 
    Monte Carlo simulation run for n times (assigned in number_of_simulations) for each prevalence vector with 150 prevalences.
    Plots effect in accuracy, f1-score, sensitivity, specificity and positive predictive value.

    Parameters
    ----------
    number_of_simulations
        number of simulations in int

    """
    #monte carlo simulations for lab case
    ppv_plots_lab= []
    accuracy_plots_lab = []
    f1_score_plots_lab = []
    sens_plots_lab = []
    spec_plots_lab = []
    for i in range(number_of_simulations):
        ppvs, sens, spec, f1_score, accuracy = run_monte_carlo(prevalence, args_lab)
        ppv_plots_lab.append(ppvs)
        f1_score_plots_lab.append(f1_score)
        accuracy_plots_lab.append(accuracy)
        sens_plots_lab.append(sens)
        spec_plots_lab.append(spec)

    #monte carlo simulations for real world case
    ppv_plots_real = []
    accuracy_plots_real = []
    f1_score_plots_real = []
    sens_plots_real = []
    spec_plots_real = []
    for i in range(number_of_simulations):
        ppvs, sens, spec, f1_score, accuracy = run_monte_carlo(prevalence, args_real)
        ppv_plots_real.append(ppvs)
        f1_score_plots_real.append(f1_score)
        accuracy_plots_real.append(accuracy)
        sens_plots_real.append(sens)
        spec_plots_real.append(spec)
    
    plotting(prevalence, ppv_plots_lab, f1_score_plots_lab, accuracy_plots_lab, sens_plots_lab, spec_plots_lab,
             ppv_plots_real, f1_score_plots_real, accuracy_plots_real, sens_plots_real, spec_plots_real)


monte_carlo(1000)
