import matplotlib.pyplot as plt
plt.style.use('./article.mplstyle')
import model_functions
from monte_carlo_model import run_monte_carlo
from utils import load_yaml
from plotting import plotting_sensitivity_analysis

#loading the arguments from yaml files
args_sens_dis_mim = load_yaml('experiments/experiment_sens_dis_mim.yaml')
args_sens_noise = load_yaml('experiments/experiment_sens_noise.yaml')
args_sens_fpr = load_yaml('experiments/experiment_sens_fpr.yaml')

#generating prevalence and calculating indices of specific prevalence
prevalence = model_functions.generate_prevalence()

def monte_carlo(number_of_simulations = 5):
    #monte carlo simulations for the sensitivity analysis
    ppv_plots_disease_mimics= []

    for i in range(number_of_simulations):
        ppvs, _, _, _, _ = run_monte_carlo(prevalence, args_sens_dis_mim)
        ppv_plots_disease_mimics.append(ppvs)

    ppv_plots_noise= []
    for i in range(number_of_simulations):
        ppvs, _, _, _, _ = run_monte_carlo(prevalence, args_sens_noise)
        ppv_plots_noise.append(ppvs)


    ppv_plots_fpr= []
    for i in range(number_of_simulations):
        ppvs, _, _, _, _ = run_monte_carlo(prevalence, args_sens_fpr)
        ppv_plots_fpr.append(ppvs)
    
    plotting_sensitivity_analysis(prevalence, ppv_plots_disease_mimics, ppv_plots_noise, ppv_plots_fpr)


monte_carlo(1000)
