import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./article.mplstyle')
import model_functions
import random

def run_monte_carlo(prevalence: np.ndarray, args):
    """
    Simulation done for each prevalance
    Parameters
    ----------
    prevalence
        Array with 150 prevalences
    args
        Probabilities for the simulation
    Returns
    ----------
    ppvs
        Array of positive predictive values
    sensitivity
        Array of sensitivities
    specificity
        Array of specificity
    f1_score
        Array of F1-scores
    accuracy
        Array of accuracies
    """
    n = 10000
    ppvs = []
    f1_score = []
    accuracy = []
    sensitivity = []
    specificity = []

    for prev in prevalence:
        #creating true label and prediction vectors with specific prevalence (predictions is copy of the same vector at this point)
        labels, predictions = model_functions.create_labels(prev, n)
        #from the values given in the arguments, the probability of true positives and false positives (transition matrix) are calculated
        transition_mat = model_functions.calculate_probabilities(args)
        #the predictions are modified based on the above probabilities
        predictions = model_functions.modify_predictions(predictions, transition_mat)
        #accuracy, f1-score, sensitivity, specificity and ppv are calculated
        sens = model_functions.calculate_sensitivity(labels, predictions)
        spec = model_functions.calculate_specificity(labels, predictions)
        sensitivity.append(sens)
        specificity.append(spec)
        ppvs.append(model_functions.calculate_ppv(sens, spec, prev))
        f1_score.append(model_functions.calculate_f1score(labels, predictions))
        accuracy.append(model_functions.calculate_accuracy(labels, predictions))

    return ppvs, sensitivity, specificity, f1_score, accuracy


def run_monte_carlo_longitudinal(predictions_all_long, labels_all_long, prevalence, args_real, args_no_noise, n):
    predictions_modified = []
    predictions_modified_wo_noise = []   
    predictions_one = []
    labels_all_one = []

    #modify predictions
    for predictions in predictions_all_long:
        transition_mat = model_functions.calculate_probabilities(args_real)
        #the predictions are modified based on the above probabilities
        predictions_mod = model_functions.modify_predictions(predictions, transition_mat)
        predictions_modified.append(predictions_mod)

    #modify predictions without noise
    for predictions in predictions_all_long:
        transition_mat = model_functions.calculate_probabilities(args_no_noise)
        #the predictions are modified based on the above probabilities
        predictions_mod = model_functions.modify_predictions(predictions, transition_mat)
        predictions_modified_wo_noise.append(predictions_mod)

    #long recording to one label
    for label in labels_all_long:
        if np.sum(label) == 0:
            labels_all_one.append(0)
        else:
            labels_all_one.append(1)

    #apply 5-of-6 post-processing rule
    sens_56, fpr_56, ppv_56, positive_proportion_56 = model_functions.rule_5_of_6(predictions_modified, labels_all_one, prevalence, n)
    sens_56_noise, fpr_56_noise, ppv_56_noise, positive_proportion_56_noise = model_functions.rule_5_of_6(predictions_modified_wo_noise, labels_all_one, prevalence, n)

    #long recording to one label
    for pred_mod in predictions_modified:
        if np.sum(pred_mod) == 0:
            predictions_one.append(0)
        else:
            predictions_one.append(1)


    sens_one = model_functions.calculate_sensitivity(labels_all_one, predictions_one)
    spec_one = model_functions.calculate_specificity(labels_all_one, predictions_one)
    fpr_one = 1-model_functions.calculate_specificity(labels_all_one, predictions_one)
    ppv_one = model_functions.calculate_ppv(sens_one, spec_one, prevalence)
    positive_proportion_one = np.sum(predictions_one)/n


    sens_rules = []
    fpr_rules = []
    ppv_rules = []
    positive_proportion = []

    #apply positive-segment ratio rule
    for j in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        predictions_one_per_rec = []

        for pred_mod in predictions_modified:
            if j == 0:
                if np.sum(pred_mod) == 0:
                    predictions_one_per_rec.append(0)
                else:
                    predictions_one_per_rec.append(1)
            else:
                if np.sum(pred_mod) < j*len(pred_mod):
                    predictions_one_per_rec.append(0)
                else:
                    predictions_one_per_rec.append(1)



        sens_rules.append(model_functions.calculate_sensitivity(labels_all_one, predictions_one_per_rec))
        fpr_rules.append(1-model_functions.calculate_specificity(labels_all_one, predictions_one_per_rec))
        sens = model_functions.calculate_sensitivity(labels_all_one, predictions_one_per_rec)
        spec = model_functions.calculate_specificity(labels_all_one, predictions_one_per_rec)
        try:
            ppv_rules.append(model_functions.calculate_ppv(sens, spec, prevalence))
        except:
            ppv_rules.append(np.nan)
        positive_proportion.append(np.sum(predictions_one_per_rec)/n)

    return sens_56, fpr_56, ppv_56, positive_proportion_56, sens_56_noise, fpr_56_noise, ppv_56_noise, positive_proportion_56_noise, sens_one, fpr_one, ppv_one, positive_proportion_one, sens_rules, fpr_rules, ppv_rules, positive_proportion


