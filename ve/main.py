import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = "device=cpu,floatX=float64,openmp=True,force_device=True"
os.environ['OMP_NUM_THREADS'] = '16'

import csv

from gaydhani import gaydhani
from malmasi import malmasi
from mondal import mondal
from zhang import zhang
from badjatiya import badjatiya
import sklearn
import sklearn.metrics as metrics
from statistics import mean, stdev
import numpy as np
from datetime import datetime
import gensim.downloader as api

class PrecisionRecallF1:

    def __init__(self, precision, recall, f1):

        self.precision = MeanStandardDeviation(precision)
        self.recall = MeanStandardDeviation(recall)
        self.f1 = MeanStandardDeviation(f1)

    def scale(self, precision_scale_factor, recall_scale_factor, f1_scale_factor):

        return PrecisionRecallF1([self.precision.scale(precision_scale_factor)],
                                 [self.recall.scale(recall_scale_factor)],
                                 [self.f1.scale(f1_scale_factor)])

    def __repr__(self):

        return str(self.precision) + ',' + str(self.recall) + ',' + str(self.f1)


class MeanStandardDeviation:

    def __init__(self, list):

        if len(list) == 1:
            self.mean = list[0]
            self.standard_deviation = None
        else:
            self.mean = mean(list)
            self.standard_deviation = stdev(list)

    def scale(self, factor):

        return self.mean * factor

    def __repr__(self):
        if self.standard_deviation is None:
            return str(percent(self.mean)) + ',{}'
        else:
            return str(percent(self.mean)) + ',{' + str(percent(self.standard_deviation)) + '}'

class Evaluation:

    def __init__(self,
                 name,
                 labels,
                 per_label_metrics,
                 micro_mean:PrecisionRecallF1,
                 macro_mean:PrecisionRecallF1,
                 weighted_mean:PrecisionRecallF1,
                 confusion_matrix_mean,
                 confusion_matrix_ratios):

        self.labels = labels
        self.name = name
        self.per_label_metrics = per_label_metrics
        self.micro_mean = micro_mean
        self.macro_mean = macro_mean
        self.weighted_mean = weighted_mean
        self.confusion_matrix_mean = confusion_matrix_mean
        self.confusion_matrix_ratios = confusion_matrix_ratios

    def print_label(self, target_label):

        for label, result in self.per_label_metrics:

            if target_label == label:
                print(self.name + ',' + str(result))

    def print_aggregate(self):

        print(self.name + ',micro/accuracy,' + str(self.micro_mean))

        print(self.name + ',macro,' + str(self.macro_mean))

        print(self.name + ',weighted macro,' + str(self.weighted_mean))


    def print_all(self):

        print(self.name, '- metrics')

        print('metric,precision,sig,recall,sig,f1,sig')

        for label, result in self.per_label_metrics:

            print('{' + label + '},' + str(result))

        print('macro,' + str(self.macro_mean))

        print('weighted macro,' + str(self.weighted_mean))

        print('micro/accuracy,' + str(self.micro_mean))

        self.print_confusion_matrix(mean=self.confusion_matrix_mean, ratios=self.confusion_matrix_ratios, labels=self.labels)

    def print_confusion_matrix(self, mean, ratios, labels):

        if len(mean) == 0 or len(ratios) == 0:
            return

        print(self.name, '- confusion matrix')

        print(',' + ' #,sig,%,sig,'.join(labels) + ' #,sig,%,sig')

        for i, label in enumerate(labels):

            mean_values = mean[i]
            ratio_values = ratios[i]

            line = []

            for j in range(len(labels)):

                line.append(str(mean_values[j]))
                line.append(str(ratio_values[j]))

            print(label + ',' + ','.join(line))

    def scale(self, precision_scale_factor, recall_scale_factor, f1_scale_factor):

        print('scaling', self.name)
        print(',precision,recall,f1')
        print('ratio,' + percent(precision_scale_factor), percent(recall_scale_factor), percent(f1_scale_factor))
        print('%change,' + percent((precision_scale_factor-1)*100, dp=2), percent((recall_scale_factor-1) * 100, dp=2), percent((f1_scale_factor-1) * 100, dp=2))

        scaled_label_metrics = []

        for label, metric in self.per_label_metrics:
            scaled_label_metrics.append((label, metric.scale(precision_scale_factor, recall_scale_factor, f1_scale_factor)))

        return Evaluation(name= self.name + ' Scaled',
                          labels=self.labels,
                          per_label_metrics=scaled_label_metrics,
                          micro_mean=self.micro_mean.scale(precision_scale_factor, recall_scale_factor, f1_scale_factor),
                          macro_mean=self.macro_mean.scale(precision_scale_factor, recall_scale_factor, f1_scale_factor),
                          weighted_mean=self.weighted_mean.scale(precision_scale_factor, recall_scale_factor, f1_scale_factor),
                          confusion_matrix_mean=[],
                          confusion_matrix_ratios=[])

class Model:

    def __init__(self, model):

        name, classifier, tweets = model

        self.name = name
        self.classifier = classifier
        self.tweets = tweets

INDENT = '    '

def main():

    print(api.load("word2vec-google-news-300", return_path=True))
    print(api.load("glove-twitter-25", return_path=True))
    return

    print('Started:', datetime.now())

    folds = 10
    seed = 314

    hateful_label = 'hateful'
    abusive_label = 'abusive'

    labels = [
        hateful_label,
        abusive_label
    ]

    default_label = 'neither'

    raw_data = load_tweets('founta-dataset.csv', labels, default_label)

    labels.append(default_label)

    total_length = len(raw_data)

    print('dataset distribution')

    for label in labels:

        count = len([t for t, l in raw_data if l == label])

        print('{' + label + '},' + str(count) + ',' + percent((count / total_length) * 100, dp=2))

    print('total,' + str(total_length))

    baseline_combined_label = abusive_label + ' or ' + default_label

    mondal_model = mondal(raw_data=raw_data, target_label=hateful_label, fallback_label=baseline_combined_label, all_original_labels=labels)

    mondal_results = evaluate_baseline(model=mondal_model, labels=[hateful_label, baseline_combined_label])

    mondal_results.print_all()

    print('Number of k-folds:', folds)
    print('Random seed:', seed)

    gaydhani_model= Model(gaydhani(raw_data))

    gaydhani_results = evaluate_model(model=gaydhani_model, labels=labels, folds=folds, seed=seed)

    gaydhani_results.print_all()

    malmasi_model = Model(malmasi(raw_data))

    malmasi_results = evaluate_model(model=malmasi_model, labels=labels, folds=folds, seed=seed)

    malmasi_results.print_all()

    zhang_model = Model(zhang(raw_data))

    zhang_results = evaluate_model(model=zhang_model, labels=labels, folds=folds, seed=seed)

    zhang_results.print_all()

    badjatiya_model = Model(badjatiya(raw_data=raw_data, labels=labels))

    badjatiya_results = evaluate_model(model=badjatiya_model, labels=labels, folds=folds, seed=seed)

    badjatiya_results.print_all()

    #Scaled results based on performance drop observed in Arango et al
    precision_scale_factor = mean([82.3/94.6, 81.6/93.7])

    recall_scale_factor = mean([82.1/94.6, 68.9/92.6])

    f1_scale_factor = mean([80.7/94.6, 73.1/93.1])

    badjatiya_original = 0.930

    print(badjatiya_model.name, 'original weighted-macro results + scaled')
    print('precision,recall,f1')
    print('original,' + percent(badjatiya_original) + ',' + percent(badjatiya_original) + ',' + percent(badjatiya_original))
    print('scaled,' + percent(badjatiya_original * precision_scale_factor) + ',' + percent(badjatiya_original * recall_scale_factor) + ',' + percent(badjatiya_original * f1_scale_factor))

    scaled_badjatiya_results = badjatiya_results.scale(precision_scale_factor,recall_scale_factor,f1_scale_factor)

    scaled_badjatiya_results.print_all()

    print('aggregate comparison')
    print('model,metric,precision,recall,f1')
    gaydhani_results.print_aggregate()
    malmasi_results.print_aggregate()
    zhang_results.print_aggregate()
    scaled_badjatiya_results.print_aggregate()

    print('{' + hateful_label + '} comparison')
    print('model,precision,recall,f1')
    gaydhani_results.print_label(hateful_label)
    malmasi_results.print_label(hateful_label)
    zhang_results.print_label(hateful_label)
    scaled_badjatiya_results.print_label(hateful_label)

    print('Completed:', datetime.now())


def evaluate_baseline(model, labels):

    name, predictions, truth = model

    print('Evaluating', name + '...')

    all, micro, macro, weighted, confusion_matrix, confusion_ratios = evaluate_fold(predictions=predictions, truth=truth, labels=labels)

    results = calculate_means(name=name, all=[all], micro=[micro], macro=[macro], weighted=[weighted], confusion_matricies=[confusion_matrix], confusion_ratios=[confusion_ratios], labels=labels)

    print('Finished evaluating', name)

    return results


def evaluate_model(model:Model, labels, folds, seed):

    print('Evaluating', model.name + '...')

    splitter = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    all = []
    micro = []
    macro = []
    weighted = []
    confusion_matricies = []
    confusion_ratios = []

    X = np.array([x for x, y in model.tweets])
    y = np.array([y for x, y in model.tweets])

    i = 1

    classifier = model.classifier

    for train_index, test_index in splitter.split(X, y):
        print(INDENT, str(i) + 'th fold...')

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(INDENT, INDENT, 'Fitting...')

        classifier.fit(X_train, y_train)

        print(INDENT, INDENT, 'Testing...')

        predictions = classifier.predict(X_test)

        print(INDENT, INDENT, 'Evaluating...')

        fold_metrics = evaluate_fold(predictions=predictions, truth=y_test, labels=labels)

        fold_all, fold_micro, fold_macro, fold_weighted, fold_cm, fold_cr = fold_metrics

        all.append(fold_all)
        micro.append(fold_micro)
        macro.append(fold_macro)
        weighted.append(fold_weighted)
        confusion_matricies.append(fold_cm)
        confusion_ratios.append(fold_cr)

        i = i + 1

    model_results = calculate_means(name=model.name, all=all, micro=micro, macro=macro, weighted=weighted, confusion_matricies=confusion_matricies, confusion_ratios=confusion_ratios, labels=labels)

    print('Finished evaluating', model.name)

    return model_results

def evaluate_fold(predictions, truth, labels):

    all = evaluate_metrics_for_average(predictions, truth, labels, 'all')
    micro = evaluate_metrics_for_average(predictions, truth, labels, 'micro')
    macro = evaluate_metrics_for_average(predictions, truth, labels, 'macro')
    weighted = evaluate_metrics_for_average(predictions, truth, labels, 'weighted')
    confusion_matrix = metrics.confusion_matrix(y_true=truth, y_pred=predictions, labels=labels)
    confusion_ratios= metrics.confusion_matrix(y_true=truth, y_pred=predictions, labels=labels, normalize='true')

    return all, micro, macro, weighted, confusion_matrix, confusion_ratios

def calculate_means(name, all, micro, macro, weighted, confusion_matricies, confusion_ratios, labels):

    per_label = calculate_mean_for_label_fold_results(all, labels)

    micro_mean = calculate_mean_for_fold_results(micro)

    macro_mean = calculate_mean_for_fold_results(macro)

    weighted_mean = calculate_mean_for_fold_results(weighted)

    mean = calculate_mean_confusion_matrix(confusion_matricies=confusion_matricies, labels=labels)

    ratios = calculate_mean_confusion_matrix(confusion_matricies=confusion_ratios, labels=labels)

    return Evaluation(name= name,
                      labels=labels,
                      per_label_metrics=per_label,
                      micro_mean=micro_mean,
                      macro_mean=macro_mean,
                      weighted_mean=weighted_mean,
                      confusion_matrix_mean=mean,
                      confusion_matrix_ratios=ratios)


def calculate_mean_confusion_matrix(confusion_matricies, labels):

    final_matrix = []

    for _ in labels:

        inner = []

        for _ in labels:

            inner.append([])

        final_matrix.append(inner)

    for fold in confusion_matricies:

        for outer in range(len(fold)):

            for inner in range(len(fold[outer])):

                final_matrix[outer][inner].append(fold[outer][inner])

    for outer in range(len(final_matrix)):

        for inner in range(len(final_matrix[outer])):

            final_matrix[outer][inner] = MeanStandardDeviation(final_matrix[outer][inner])

    return final_matrix

def percent(value, dp = 3):

    parts = str(round(value, dp)).split('.')

    int = parts[0]
    dec = '0'

    if len(parts) == 2:
        dec = parts[1]

    return int  + '.' + dec.ljust(dp, '0')

def calculate_mean_for_label_fold_results(kfold_results, labels):

    all_precision = {}
    all_recall = {}
    all_f1 = {}

    for label in labels:
        all_precision[label] = []
        all_f1[label] = []
        all_recall[label] = []

    for precision, recall, f1 in kfold_results:

        for i, label in enumerate(labels):

            all_precision[label].append(precision[i])
            all_recall[label].append(recall[i])
            all_f1[label].append(f1[i])

    final_results = []

    for label in labels:

        final_results.append((label, PrecisionRecallF1(all_precision[label], all_recall[label], all_f1[label])))

    return final_results

def calculate_mean_for_fold_results(kfold_results):

    precision = [precision for precision, _, _ in kfold_results]
    recall = [recall for _, recall, _ in kfold_results]
    f1 = [f1 for _, _, f1 in kfold_results]

    return PrecisionRecallF1(precision, recall, f1)

def evaluate_metrics_for_average(predictions, test_labels, all_labels, metric):

    final_metric = metric;

    if metric == 'all':
        final_metric = None

    precision = metrics.precision_score(y_true=test_labels, y_pred=predictions, labels=all_labels, average=final_metric)
    recall = metrics.recall_score(y_true=test_labels, y_pred=predictions, labels=all_labels, average=final_metric)
    f1 = metrics.f1_score(y_true=test_labels, y_pred=predictions, labels=all_labels, average=final_metric)

    return precision, recall, f1

def load_tweets(data_set_name, target_labels, default_label):

    raw_data = []

    with open(data_set_name, newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')

        for row in reader:

            assert(len(row) == 3)

            text, label, confidence = row

            if text == 'text':
                #Skip the header row
                continue;

            assert (len(text.strip()) > 0)
            assert (len(label.strip()) > 0)

            label = label.lower()

            if not label in target_labels:
                raw_data.append([text, default_label])
            else:
                raw_data.append([text, label])

    return raw_data



if __name__ == "__main__":
    main()

