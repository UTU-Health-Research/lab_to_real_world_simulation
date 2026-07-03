import numpy as np
import sklearn

def calculate_ppv(sensitivity, specificity, prevalence):
    ppv = (sensitivity*prevalence)/((sensitivity*prevalence)+((1-specificity)*(1-prevalence)))

    return ppv

def generate_prevalence():
    prevalence = np.linspace(0.001, 0.999, 150)

    return prevalence


def calculate_sensitivity(labels, predictions):
    sensitivity = sklearn.metrics.recall_score(labels, predictions)

    return sensitivity

def calculate_specificity(labels, predictions):
    specificity = sklearn.metrics.recall_score(labels, predictions, pos_label=0)

    return specificity

def calculate_accuracy(labels, predictions):
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)

    return accuracy

def calculate_f1score(labels, predictions):
    f1 = sklearn.metrics.f1_score(labels, predictions)

    return f1

def create_labels(prevalence, n):
    #labels created by randomly assigning values to 0 or 1 with the probability of the prevalence
    labels = np.random.binomial(1, prevalence, size=n)
    predictions = labels.copy()

    return labels, predictions

def modify_predictions(predictions, transition_mat):
    for i, pred in enumerate(predictions):
        if pred == 0:
            predictions[i] = np.random.binomial(1, transition_mat[0, 1])
        else:
            predictions[i] = np.random.binomial(1, transition_mat[1,1])
            
    return predictions

def calculate_probabilities(args):
    #calculates probabilities that the predicition is positive when true label is positive and the prediction is positive when true label is negative
    #probabilities randomly drawn from the given distribution
    #probabilities to cause false positive
    disease_mimics_prev_fp = np.random.uniform(args.disease_mimic_prevalence_fp[0], args.disease_mimic_prevalence_fp[1])
    disease_mimics_prob_fp = np.random.normal(args.disease_mimic_prob_fp[0], args.disease_mimic_prob_fp[1])
    if disease_mimics_prob_fp < 0:
        disease_mimics_prob_fp = 0
    if disease_mimics_prob_fp > 1:
        disease_mimics_prob_fp = 1
    fpr_fp = np.random.normal(args.fpr_fp[0], args.fpr_fp[1])
    if fpr_fp < 0:
        fpr_fp = 0
    if fpr_fp > 1:
        fpr_fp = 1
    #probabilities to cause false negatives
    disease_mimics_prev_fn = np.random.uniform(args.disease_mimic_prevalence_fn[0], args.disease_mimic_prevalence_fn[1])
    disease_mimics_prob_fn = np.random.normal(args.disease_mimic_prob_fn[0], args.disease_mimic_prob_fn[1])
    fnr_fn = np.random.normal(args.fnr_fn[0], args.fnr_fn[1])
    if fnr_fn < 0:
        fnr_fn = 0
    if fnr_fn > 1:
        fnr_fn = 1
    #probabilities for noise
    if args.lab == False:
        #for real
        noise_prob_fp = np.random.beta(args.noise_prob_fp[0], args.noise_prob_fp[1])
        noise_prob_fn = np.random.normal(args.noise_prob_fn[0], args.noise_prob_fn[1])
    else:
        #for lab
        noise_prob_fp = np.random.normal(args.noise_prob_fp[0], args.noise_prob_fp[1])
        noise_prob_fn = np.random.normal(args.noise_prob_fn[0], args.noise_prob_fn[1])

    if noise_prob_fn < 0:
        noise_prob_fn = 0
    if noise_prob_fn > 1:
        noise_prob_fn = 1
    if noise_prob_fp < 0:
        noise_prob_fp = 0
    if noise_prob_fp > 1:
        noise_prob_fp = 1

    #form transition matrices for each factor
    dm_mat = [[(1-disease_mimics_prev_fp*disease_mimics_prob_fp), disease_mimics_prev_fp*disease_mimics_prob_fp], [disease_mimics_prev_fn*disease_mimics_prob_fn, (1-disease_mimics_prev_fn*disease_mimics_prob_fn)]]
    noise_mat = [[(1-noise_prob_fp), noise_prob_fp], [noise_prob_fn, (1-noise_prob_fn)]]
    fpr_fnr_mat = [[(1-fpr_fp), fpr_fp], [fnr_fn, (1-fnr_fn)]]

    #calculate the final transition matrix
    transition_mat = np.matmul(np.matmul(fpr_fnr_mat, dm_mat), noise_mat)

    return transition_mat



