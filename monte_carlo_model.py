import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./article.mplstyle')
import model_functions


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


