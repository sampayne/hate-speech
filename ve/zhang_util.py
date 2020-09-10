import csv
import pickle
import os
import numpy as np
import datetime
import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

import zhang_text_preprocess

def load_classifier_model(classifier_pickled=None):
    if classifier_pickled:
        with open(classifier_pickled, 'rb') as model:
            classifier = pickle.load(model)
        return classifier

def outputFalsePredictions(pred, truth, index, model_descriptor, dataset_name, outfolder):
    return

def prepare_score_string(p, r, f1, s, labels, target_names, digits):
    string = ",precision,recall,f1,support\n"
    for i, label in enumerate(labels):
        string = string + target_names[i] + ","
        for v in (p[i], r[i], f1[i]):
            string = string + "{0:0.{1}f}".format(v, digits) + ","
        string = string + "{0}".format(s[i]) + "\n"
        # values += ["{0}".format(s[i])]
        # report += fmt % tuple(values)

    return string


def save_scores(nfold_texts,
                nfold_predictions,
                nfold_truth,
                heldout_texts,
                heldout_predictions,
                heldout_truth,
                #nfold_index,
                #heldout_index,
                model_descriptor,
                #dataset_name,
                digits,
                #outfolder,
                instance_tags_train= None,
                instance_tags_test= None,
                accepted_ds_tags: list = None):

    #pred = nfold_predictions.tolist()+heldout_predictions.tolist()
    #truth = list(nfold_truth)+list(heldout_truth)
    #index = nfold_index+heldout_index

    #outputFalsePredictions(pred, truth, index, model_descriptor,dataset_name, outfolder)

    #subfolder = outfolder + "/scores"

    #try:
    #    os.stat(subfolder)
    #except:
    #    os.mkdir(subfolder)

    #filename = os.path.join(subfolder, "SCORES_%s.csv" % (dataset_name))
    writer = None#writer = open(filename, "a+")
    #writer.write(model_descriptor+"\n")
    print(model_descriptor)

    if nfold_predictions is not None:

        #writer.write(" N-FOLD AVERAGE :\n")
        print(" N-FOLD AVERAGE :")

        write_scores(texts=nfold_texts,
                     predictoins=nfold_predictions,
                     truth=nfold_truth,
                     digits=digits,
                     writer=writer,
                     instance_dst_column=instance_tags_train,
                     accepted_ds_tags=accepted_ds_tags)

    if (heldout_predictions is not None):

        print(" HELDOUT :")

        write_scores(texts=heldout_texts,
                     predictoins=heldout_predictions,
                     truth=heldout_truth,
                     digits=digits,
                     writer=writer,
                     instance_dst_column=instance_tags_test,
                     accepted_ds_tags=accepted_ds_tags)

    #writer.close()

def data_stats(X_train_data, y_train):
    labels={}
    for y in y_train:
        if y in labels.keys():
            labels[y]+=1
        else:
            labels[y]=1
    print("instances={}, labels={}, label distribution={}".
          format(len(X_train_data), len(y_train),labels))

def write_scores(texts,
                 predictoins,
                 truth,
                 digits,
                 writer,
                 instance_dst_column=None,
                 accepted_ds_tags=None):

    data_stats(texts, truth)

    labels = unique_labels(truth, predictoins)

    print('unique_labels:', labels)

    if accepted_ds_tags is None:
        target_names = ['%s' % l for l in labels]
        p, r, f1, s = precision_recall_fscore_support(truth, predictoins,
                                                      labels=labels)

        line = prepare_score_string(p, r, f1, s, labels, target_names, digits)
        pa, ra, f1a, sa = precision_recall_fscore_support(truth, predictoins,
                                                          average='micro')
        line += "avg_micro,"
        for v in (pa, ra, f1a):
            line += "{0:0.{1}f}".format(v, digits) + ","
        line += '{0}'.format(np.sum(sa)) + "\n"
        pa, ra, f1a, sa = precision_recall_fscore_support(truth, predictoins,
                                                          average='macro')
        line += "avg_macro,"
        for v in (pa, ra, f1a):
            line += "{0:0.{1}f}".format(v, digits) + ","
        line += '{0}'.format(np.sum(sa)) + "\n\n"
        # average

        #writer.write(line)
        print(line)

    if accepted_ds_tags is not None:

        for dstag in accepted_ds_tags:
            #writer.write("\n for data from {} :\n".format(dstag))
            subset_pred = []
            subset_truth = []

            for ds, label in zip(instance_dst_column, predictoins):
                if ds == dstag:
                    if isinstance(label, np.ndarray):
                        subset_pred.append(label[0])
                    else:
                        subset_pred.append(label)

            for ds, label in zip(instance_dst_column, truth):
                if ds == dstag:
                    subset_truth.append(label)
            # print("subset_truth={}, type={}".format(len(subset_truth), type(subset_truth)))
            # print("subset_pred={}, type={}".format(len(subset_pred), type(subset_pred)))
            subset_labels = unique_labels(subset_truth, subset_pred)
            target_names = ['%s' % l for l in labels]
            p, r, f1, s = precision_recall_fscore_support(subset_truth, subset_pred,
                                                          labels=subset_labels)

            line = prepare_score_string(p, r, f1, s, labels, target_names, digits)
            pa, ra, f1a, sa = precision_recall_fscore_support(subset_truth, subset_pred,
                                                              average='micro')
            line += "avg_micro,"
            for v in (pa, ra, f1a):
                line += "{0:0.{1}f}".format(v, digits) + ","
            line += '{0}'.format(np.sum(sa)) + "\n"
            pa, ra, f1a, sa = precision_recall_fscore_support(subset_truth, subset_pred,
                                                              average='macro')
            line += "avg_macro,"
            for v in (pa, ra, f1a):
                line += "{0:0.{1}f}".format(v, digits) + ","
            line += '{0}'.format(np.sum(sa)) + "\n\n"

            #writer.write(line)
            print(line)


def index_max(values):
    return max(range(len(values)), key=values.__getitem__)


def save_classifier_model(model, outfile):
    if model:
        with open(outfile, 'wb') as model_file:
            pickle.dump(model, model_file)


def validate_training_set(training_set):
    """
    validate training data set (i.e., X) before scaling, PCA, etc.
    :param training_set: training set, test data
    :return:
    """
    # print("np any isnan(X): ", np.any(np.isnan(training_set)))
    # print("np all isfinite: ", np.all(np.isfinite(training_set)))
    # check any NaN row
    row_i = 0
    for i in training_set:
        row_i += 1
        if np.any(np.isnan(i)):
            print("ERROR: [", row_i, "] is nan: ")
            print(i)


def feature_scaling_mean_std(feature_set):
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(feature_set)


def feature_scaling_min_max(feature_set):
    """
    Input X must be non-negative for multinomial Naive Bayes model
    :param feature_set:
    :return:
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(feature_set)


def class_to_name(class_label):
    """
    This function can be used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "Hate speech"
    elif class_label == 1:
        return "Offensive language"
    elif class_label == 2:
        return "Neither"
    else:
        return "No label"


def name_to_class(class_label):
    # U= unknown, R = Religion, E = Ethnicity, S = Sexuality, Y = yes blank = no, x = don't use
    if (class_label == "r") or (class_label == "e") or (class_label == "s") or (class_label == "y"):
        return "0"  # Hate speech
    elif class_label == "":
        return "2"  # neither
    else:
        return "x"  # dont use


def output_data_splits(data_file, out_folder):
    raw_data = pd.read_csv(data_file, sep=',', encoding="utf-8")
    X_train_data, X_test_data, y_train, y_test = \
        train_test_split(raw_data, raw_data['class'],
                         test_size=0.5,
                         random_state=42)
    X_train_data.to_csv(out_folder + "/part1.csv", sep=',', encoding='utf-8')
    X_test_data.to_csv(out_folder + "/part2.csv", sep=',', encoding='utf-8')
    # raw_data = pd.read_csv(data_file, sep=',', encoding="utf-8")
    # X_train_data, X_test_data, y_train, y_test = \
    #     train_test_split(raw_data, raw_data['class'],
    #                      test_size=0.33,
    #                      random_state=42)
    # X_train_1, X_test_1, y_train1, y_test1 = \
    #     train_test_split(X_train_data, X_train_data['class'],
    #                      test_size=0.50,
    #                      random_state=42)
    # X_test_data.to_csv(out_folder + "/part1.csv", sep=',', encoding='utf-8')
    # X_train_1.to_csv(out_folder + "/part2.csv", sep=',', encoding='utf-8')
    # X_test_1.to_csv(out_folder + "/part3.csv", sep=',', encoding='utf-8')


def save_selected_features(finalFeatureIndices, featureTypes, file):
    with open(file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        start = 0
        featureTypesAsDict = {}
        # for item in featureTypes:
        #     if isinstance(item, tuple):
        #         featureTypesAsDict[item[0]]=item[1]
        #     elif isinstance(item, list):
        #         i = iter(item)


        for ft_key, ft_value in featureTypes.items():
            if isinstance(ft_value[1], dict):
                feature_lookup = {v: k for k, v in ft_value[1].items()}
            else:
                feature_lookup = {v: k for v, k in enumerate(ft_value[1])}
            max = start + len(feature_lookup)
            for i in finalFeatureIndices:
                if i < start:
                    continue
                if i < max:
                    feature = feature_lookup[i - start]
                    writer.writerow([i, ft_key, feature])
            start = max

    return None


def saveOutput(prediction, model_name, task, outfolder):
    filename = os.path.join(outfolder, "prediction-%s-%s.csv" % (model_name, task))
    file = open(filename, "w")
    for entry in prediction:
        if (isinstance(entry, float)):
            file.write(str(entry) + "\n")
            # file.write("\n")
        else:
            if (entry[0] > entry[1]):
                file.write("0\n")
            else:
                file.write("1\n")
    file.close()


def feature_scale(option, M):
    if option == -1:
        return M

    print("feature scaling, first perform sanity check...")
    if not isinstance(M, np.ndarray) and M.isnull().values.any():
        print("input matrix has NaN values, replace with 0")
        M.fillna(0)

    # if self.feature_selection:
    #     print("FEATURE SELECTION BEGINS, {}".format(datetime.datetime.now()))
    #     select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
    #     M = select.fit_transform(M, self.raw_data['class'])
    #     print("REDUCED FEATURE MATRIX dimensions={}".format(M.shape))
    # if not self.feature_selection:
    # logger.logger.info("APPLYING FEATURE SCALING: [%s]" % option)
    if option == 0:  # mean std
        M = feature_scaling_mean_std(M)
        if np.isnan(M).any():
            print("scaled matrix has NaN values, replace with 0")
        M = np.nan_to_num(M)
    elif option == 1:
        M = feature_scaling_min_max(M)
        if np.isnan(M).any():
            print("scaled matrix has NaN values, replace with 0")
        M = np.nan_to_num(M)
    else:
        pass

    # print("FEATURE SELECTION BEGINS, {}".format(datetime.datetime.now()))
    # select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
    # M = select.fit_transform(M, self.raw_data['class'])
    # print("REDUCED FEATURE MATRIX dimensions={}".format(M.shape))
    print("feature scaling done")
    return M


def feature_extraction(raw_data_column,feat_vectorizer, sysout, logger, cleaned_data_column=None):
    tweets = raw_data_column
    tweets = [x for x in tweets if type(x) == str]
    logger.info("FEATURE EXTRACTION AND VECTORIZATION FOR ALL data, insatance={}, {}"
                .format(len(tweets), datetime.datetime.now()))
    logger.info("\tbegin feature extraction and vectorization...")

    if cleaned_data_column is None:
        tweets_cleaned = [text_preprocess.strip_hashtags(x) for x in tweets]
    else:
        tweets_cleaned=[x for x in cleaned_data_column if type(x) == str]
    # tweets_cleaned = [text_preprocess.preprocess_clean(x, True, True) for x in tweets]
    M = feat_vectorizer.transform_inputs(tweets, tweets_cleaned, sysout, "na")
    logger.info("FEATURE MATRIX dimensions={}".format(M[0].shape))
    return M


def read_preselected_features(only_intersection, *files):
    file_with_features = []
    for file in files:
        feature_with_values = {}
        with open(file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                ft = row[1]
                value = row[2]
                if (ft in feature_with_values.keys()):
                    feature_with_values[ft].append(value)
                else:
                    values = []
                    values.append(value)
                    feature_with_values[ft] = values
            file_with_features.append(feature_with_values)

    all_fts = set()
    all_fts.update(file_with_features[0].keys())
    for i in range(1, len(file_with_features)):
        all_fts = set.intersection(all_fts, file_with_features[i].keys())

    selected_features = {}
    for ft in all_fts:
        selected = []
        for file_features in file_with_features:
            values = file_features[ft]
            selected.append(set(values))

        if only_intersection:
            selected_features[ft] = set.intersection(*selected)
        else:
            selected_features[ft] = set.union(*selected)

    return selected_features


def tag_source_file(csv_tdc_a, out_file):
    with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        with open(csv_tdc_a, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count = 0
            for row in csvreader:
                if count == 0:
                    row.insert(0, "")
                    writer.writerow(row)
                    count += 1
                    continue

                if (len(row) > 7):
                    tweet_id = row[7]
                else:
                    row.insert(0, "td")
                    writer.writerow(row)
                    continue

                try:
                    float(tweet_id)
                except ValueError:
                    if len(row) > 8:
                        tweet_id = row[8]
                    else:
                        tweet_id = ""

                if len(tweet_id) == 0:
                    row.insert(0, "td")
                else:
                    row.insert(0, "c")
                writer.writerow(row)


def balanced_tdc_mixed(td_2_c_ratio, in_csv, out_csv):
    random.sample([1, 2, 3, 4, 5], 3)
    header = None
    c_rows = []
    td_rows = []
    with open(in_csv, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        count = 0
        for row in csvreader:
            if count == 0:
                header = row
                count += 1
                continue

            if row[0] == 'c':
                c_rows.append(row)
            else:
                td_rows.append(row)

    sample_size = int(td_2_c_ratio * len(c_rows))
    td_rows = random.sample(td_rows, sample_size)
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in c_rows:
            writer.writerow(row)
        for row in td_rows:
            writer.writerow(row)


def separate_tdc(in_csv, out_csv, tag):
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        with open(in_csv, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count = 0
            for row in csvreader:
                if count == 0:
                    writer.writerow(row)
                    count += 1
                    continue

                if row[0] == tag:
                    writer.writerow(row)
                else:
                    continue


def remove_offensive_label(in_file, out_file):
    with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        with open(in_file, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count = 0
            for row in csvreader:
                if count == 0:
                    writer.writerow(row)
                    count += 1
                    continue

                # if row[6] == "1":
                #     row[6] = "2"

                if row[6] == "3":
                    continue
                # if row[6] == "1" or row[6]==3:
                #     row[6]="0"

                writer.writerow(row)

def read_word_dist_features(csv_file):
    if csv_file is None:
        return None
    with open(csv_file, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        count=0
        pos_col=-1
        result={}
        for row in csvreader:
            count+=1
            if count==1:
                for i in range(0,len(row)):
                    if row[i]=='0':
                        pos_col=i
                continue

            word=row[0]
            scores={}
            neg_sum=0
            for i in range(1, len(row)):
                if i==pos_col:
                    scores["0"]=row[i]
                else:
                    neg_sum+=float(row[i])
            scores["2"]=neg_sum
            result[word]=scores
        return result

def concat_matrices(matrix1, matrix2):
    concat = np.concatenate((matrix1,matrix2), axis=1)
    return concat

def build_word_dist_features(word, word_dist_scores_map):
    vector=[]
    if word in word_dist_scores_map.keys():
        word_dist=word_dist_scores_map[word]
        vector.append(word_dist["0"])
        vector.append(word_dist["2"])
    else:
        vector.append(0.5)
        vector.append(0.5)
    return vector
