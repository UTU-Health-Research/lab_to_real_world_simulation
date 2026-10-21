import numpy as np
from pandas import read_csv
import random
import model_functions
from utils import load_yaml
import matplotlib.pyplot as plt
from monte_carlo_model import run_monte_carlo_longitudinal

args_lab = load_yaml('experiments/experiment_lab.yaml')
args_real = load_yaml('experiments/experiment_real.yaml')
args_no_noise = load_yaml('experiments/experiment_no_noise.yaml')


prevalence = [0.052, 0.153, 0.234]

burden_prob = [0.619, 0.198, 0.183]

n = 1000
fig, ax = plt.subplots(3)
fig2, ax2 = plt.subplots(3)
for k, prev in enumerate(prevalence):
    #negative recordings
    labels_all_long = []
    predictions_all_long = []
    af_negatives = np.zeros((int(n*(1-prev)), 2880))
    for neg in af_negatives:
        labels_all_long.append(neg)
        predictions_all_long.append(neg.copy())

    #real longitudinal recordings with af segments
    longitudinal_af = read_csv('longitudinal_af.csv')

    af_low_burden, af_moderate_burden, af_high_burden = [], [], []

    #divide recordings to burden groups
    af_low_burden.append(longitudinal_af['binary_sequence_30s'][(longitudinal_af['af_burden'] > 0) & (longitudinal_af['af_burden'] < 0.02)].apply(lambda x: [int(c) for c in x]))
    af_moderate_burden.append(longitudinal_af['binary_sequence_30s'][(longitudinal_af['af_burden'] > 0.02) & (longitudinal_af['af_burden'] < 0.1)].apply(lambda x: [int(c) for c in x]))
    af_high_burden.append(longitudinal_af['binary_sequence_30s'][longitudinal_af['af_burden'] > 0.1].apply(lambda x: [int(c) for c in x]))

    af_low_burden = np.array(af_low_burden)[0]
    af_moderate_burden = np.array(af_moderate_burden)[0]
    af_high_burden = np.array(af_high_burden)[0]

    #choose burden and recording randomly
    for i in range(int(n*prev)):
        burden = random.choices([af_low_burden, af_moderate_burden, af_high_burden], burden_prob)
        af_recording = np.array(burden[0][random.choices(np.arange(len(burden[0])))[0]])
        labels_all_long.append(af_recording)
        predictions_all_long.append(af_recording.copy())


    sens_56_all, fpr_56_all, ppv_56_all, positive_proportion_56_all = [], [], [], []
    sens_56_all_noise, fpr_56_all_noise, ppv_56_all_noise, positive_proportion_56_all_noise = [], [], [], []
    sens_one_all, fpr_one_all, ppv_one_all, positive_proportion_one_all = [], [], [], []
    sens_rules_all, fpr_rules_all, ppv_rules_all, positive_proportion_rules_all = [], [], [], []


    for i in range(1000):
        #run longitudinal simulation
        sens_56, fpr_56, ppv_56, positive_proportion_56, \
        sens_56_noise, fpr_56_noise, ppv_56_noise, positive_proportion_56_noise, \
        sens_one, fpr_one, ppv_one, positive_proportion_one, sens_rules, fpr_rules, ppv_rules, positive_proportion_rules = run_monte_carlo_longitudinal(predictions_all_long, labels_all_long, prev, args_real, args_no_noise, n)

        sens_56_all.append(sens_56)
        fpr_56_all.append(fpr_56)
        ppv_56_all.append(ppv_56)
        positive_proportion_56_all.append(positive_proportion_56)
        sens_56_all_noise.append(sens_56_noise)
        fpr_56_all_noise.append(fpr_56_noise)
        ppv_56_all_noise.append(ppv_56_noise)
        positive_proportion_56_all_noise.append(positive_proportion_56_noise)
        sens_one_all.append(sens_one)
        fpr_one_all.append(fpr_one)
        ppv_one_all.append(ppv_one)
        positive_proportion_one_all.append(positive_proportion_one)
        sens_rules_all.append(sens_rules)
        fpr_rules_all.append(fpr_rules)
        ppv_rules_all.append(ppv_rules)
        positive_proportion_rules_all.append(positive_proportion_rules)


    #plotting results
    ax[k].grid(axis='y')
    ax[k].bar([1,5,9,13], [np.mean(sens_one_all), np.mean(fpr_one_all), np.mean(ppv_one_all), np.mean(positive_proportion_one_all)])
    ax[k].bar([2, 6, 10, 14], [np.mean(sens_56_all), np.mean(fpr_56_all), np.mean(ppv_56_all), np.mean(positive_proportion_56_all)])
    ax[k].bar([3, 7, 11, 15], [np.mean(sens_56_all_noise), np.mean(fpr_56_all_noise), np.mean(ppv_56_all_noise), np.mean(positive_proportion_56_all_noise)])
    ax[k].errorbar([1,5,9,13], [np.mean(sens_one_all), np.mean(fpr_one_all), np.mean(ppv_one_all), np.mean(positive_proportion_one_all)], \
                yerr=[np.std(sens_one_all), np.std(fpr_one_all), np.std(ppv_one_all), np.std(positive_proportion_one_all)], color='black', linestyle='None', capsize=3)
    ax[k].errorbar([2, 6, 10, 14], [np.mean(sens_56_all), np.mean(fpr_56_all), np.mean(ppv_56_all), np.mean(positive_proportion_56_all)], \
                yerr=[np.std(sens_56_all), np.std(fpr_56_all), np.std(ppv_56_all), np.std(positive_proportion_56_all)], color='black', linestyle='None', capsize=3)
    ax[k].errorbar([3, 7, 11, 15], [np.mean(sens_56_all_noise), np.mean(fpr_56_all_noise), np.mean(ppv_56_all_noise), np.mean(positive_proportion_56_all_noise)], \
                yerr=[np.std(sens_56_all_noise), np.std(fpr_56_all_noise), np.std(ppv_56_all_noise), np.std(positive_proportion_56_all_noise)], color='black', linestyle='None', capsize=3)
    ax[k].set_xticks([2, 6, 10, 14],['Sensitivity', 'FPR', 'PPV','Screening-positive proportion'])
    ax[k].legend(['≥ 1 positive segment', 'Adaptive 5-of-6 rule', 'Adaptive 5-of-6 rule without simulated noise'])
    ax[k].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax2[k].grid(axis='y')       
    ax2[k].plot([1,10,20,30,40,50,60,70,80,90,100], np.mean(sens_rules_all, axis=0), label='TPR')
    ax2[k].plot([1,10,20,30,40,50,60,70,80,90,100], np.mean(fpr_rules_all, axis=0), label='FPR')
    ax2[k].plot([1,10,20,30,40,50,60,70,80,90,100], np.nanmean(ppv_rules_all, axis=0), label='PPV')
    ax2[k].plot([1,10,20,30,40,50,60,70,80,90,100], np.mean(positive_proportion_rules_all, axis=0), label='Screening-positive proportion')
    ax2[k].plot([1,10,20,30,40,50,60,70,80,90,100], np.mean(sens_rules_all, axis=0), 'C0o')
    ax2[k].plot([1,10,20,30,40,50,60,70,80,90,100], np.mean(fpr_rules_all, axis=0), 'C1o')
    ax2[k].plot([1,10,20,30,40,50,60,70,80,90,100], np.nanmean(ppv_rules_all, axis=0), 'C2o')
    ax2[k].plot([1,10,20,30,40,50,60,70,80,90,100], np.mean(positive_proportion_rules_all, axis=0), 'C3o')
    ax2[k].errorbar([1,10,20,30,40,50,60,70,80,90,100], np.mean(sens_rules_all, axis=0), color='C0', yerr=np.std(sens_rules_all, axis=0), capsize=3)
    ax2[k].errorbar([1,10,20,30,40,50,60,70,80,90,100], np.mean(fpr_rules_all, axis=0), color='C1', yerr=np.std(fpr_rules_all, axis=0), capsize=3)
    ax2[k].errorbar([1,10,20,30,40,50,60,70,80,90,100], np.nanmean(ppv_rules_all, axis=0), color='C2', yerr=np.nanstd(ppv_rules_all, axis=0), capsize=3)
    ax2[k].errorbar([1,10,20,30,40,50,60,70,80,90,100], np.mean(positive_proportion_rules_all, axis=0), color='C3', yerr=np.std(positive_proportion_rules_all, axis=0), capsize=3)
    ax2[k].set_xlabel('Required positive-segment ratio (%)')
    ax2[k].legend()
    ax2[k].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.show()

