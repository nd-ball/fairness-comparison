import fire
import os
import statistics
import sys

from fairness import results
from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData
from fairness.algorithms.list import ALGORITHMS
from fairness.metrics.list import get_metrics

from fairness.algorithms.ParamGridSearch import ParamGridSearch

from sklearn.utils import resample 

NUM_TRIALS_DEFAULT = 10

def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result

def run(num_trials = NUM_TRIALS_DEFAULT, dataset = get_dataset_names(),
        algorithm = get_algorithm_names(), num_bootstrap=1):
    algorithms_to_run = algorithm

    print("Datasets: '%s'" % dataset)
    print("Bootstraps: '%s'" % num_bootstrap)

    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())

        processed_dataset = ProcessedData(dataset_obj)
        train_test_splits = processed_dataset.create_train_test_splits(num_trials)

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
        for sensitive in all_sensitive_attributes:

            print("Sensitive attribute:" + sensitive)

            detailed_files = dict((k, create_detailed_file(
                                          dataset_obj.get_results_filename(sensitive, k),
                                          dataset_obj,
                                          processed_dataset.get_sensitive_values(k), k))
                for k in train_test_splits.keys())
            
            for algorithm in ALGORITHMS:
                if not algorithm.get_name() in algorithms_to_run:
                    continue

                print("    Algorithm: %s" % algorithm.get_name())
                print("       supported types: %s" % algorithm.get_supported_data_types())
                if algorithm.__class__ is ParamGridSearch:
                    param_files =  \
                        dict((k, create_detailed_file(
                                     dataset_obj.get_param_results_filename(sensitive, k,
                                                                            algorithm.get_name()),
                                     dataset_obj, processed_dataset.get_sensitive_values(k), k))
                          for k in train_test_splits.keys())
                for i in range(0, num_trials):
                    for supported_tag in algorithm.get_supported_data_types():
                        # JL: for our exp we're only looking at numerical-binsensitive. no need to run others
                        if supported_tag != 'numerical-binsensitive':
                            continue
                        train, test = train_test_splits[supported_tag][i]
                        try:
                            params, results, param_results =  \
                                run_eval_alg(algorithm, train, test, dataset_obj, processed_dataset,
                                             all_sensitive_attributes, sensitive, supported_tag, num_bootstrap)
                        except Exception as e:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            print("Failed: %s" % e, file=sys.stderr)
                        else:
                            for j in range(len(results)):
                                write_alg_results(detailed_files[supported_tag],
                                              algorithm.get_name(), params, '{}-{}'.format(i,j), results[j]) #results)
                            #if algorithm.__class__ is ParamGridSearch:
                            #    for entry in param_results:
                            #        j = 0
                            #        for params, results in entry:
                            #            write_alg_results(param_files[supported_tag],
                            #                            algorithm.get_name(), params, '{}-{}'.format(i, j), results)
                            #            j += 1
                            #    for param_file in param_files.values():
                            #        param_file.close() 


            print("Results written to:")
            for supported_tag in algorithm.get_supported_data_types():
                print("    %s" % dataset_obj.get_results_filename(sensitive, supported_tag))

            for detailed_file in detailed_files.values():
                detailed_file.close()


def write_alg_results(file_handle, alg_name, params, run_id, results_list):
    line = alg_name + ','
    params = ";".join("%s=%s" % (k, v) for (k, v) in params.items())
    line += params + (',%s,' % run_id)
    line += ','.join(str(x) for x in results_list) + '\n'
    file_handle.write(line)

def run_eval_alg(algorithm, train, test, dataset, processed_data, all_sensitive_attributes,
                 single_sensitive, tag, num_bootstrap):
    """
    Runs the algorithm and gets the resulting metric evaluations.
    """
    privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
    positive_val = dataset.get_positive_class_val(tag)


    # JPL (08/2020): update this code so that we can bootstrap each train/test split
    # run_alg takes data frames for train and test
    # so to do a manual bootstrap I'll select indices for each (with replacement)
    # let's do 5 for now to make sure it works

    NUM_BOOTSTRAP = num_bootstrap
    trainLength = len(train)
    testLength = len(test)

    all_results = []
    all_params_lol = []
    for i in range(NUM_BOOTSTRAP):
        train_idx = resample(list(range(trainLength)), n_samples=trainLength, replace=True)
        trainDF = train.iloc[train_idx,:]
        test_idx = resample(list(range(testLength)), n_samples=testLength, replace=True)
        testDF = test.iloc[test_idx, :] 

        # get the actual classifications and sensitive attributes
        actual = testDF[dataset.get_class_attribute()].values.tolist()
        sensitive = testDF[single_sensitive].values.tolist()

        predicted, params, predictions_list =  \
            run_alg(algorithm, trainDF, testDF, dataset, all_sensitive_attributes, single_sensitive,
                    privileged_vals, positive_val)

        # make dictionary mapping sensitive names to sensitive attr test data lists
        dict_sensitive_lists = {}
        for sens in all_sensitive_attributes:
            dict_sensitive_lists[sens] = testDF[sens].values.tolist()

        sensitive_dict = processed_data.get_sensitive_values(tag)
        one_run_results = []
        for metric in get_metrics(dataset, sensitive_dict, tag):
            result = metric.calc(actual, predicted, dict_sensitive_lists, single_sensitive,
                                privileged_vals, positive_val)
            one_run_results.append(result)
        all_results.append(one_run_results)

        # handling the set of predictions returned by ParamGridSearch
        results_lol = []
        if len(predictions_list) > 0:
            for param_name, param_val, predictions in predictions_list:
                params_dict = { param_name : param_val }
                results = []
                for metric in get_metrics(dataset, sensitive_dict, tag):
                    result = metric.calc(actual, predictions, dict_sensitive_lists, single_sensitive,
                                        privileged_vals, positive_val)
                    results.append(result)
                results_lol.append( (params_dict, results) )
        all_params_lol.append(results_lol)

    #return params, one_run_results, results_lol
    return params, all_results, all_params_lol

def run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
            privileged_vals, positive_val, param_override=None):
    class_attr = dataset.get_class_attribute()
    params = algorithm.get_default_params()

    # Note: the training and test set here still include the sensitive attributes because
    # some fairness aware algorithms may need those in the dataset.  They should be removed
    # before any model training is done.
    predictions, predictions_list =  \
        algorithm.run(train, test, class_attr, positive_val, all_sensitive_attributes,
                      single_sensitive, privileged_vals, params)

    return predictions, params, predictions_list


def get_dict_sensitive_vals(dict_sensitive_lists):
    """
    Takes a dictionary mapping sensitive attributes to lists in the test data and returns a
    dictionary mapping sensitive attributes to lists containing each sensitive value only once.
    """
    newdict = {}
    for sens in dict_sensitive_lists:
         sensitive = dict_sensitive_lists[sens]
         newdict[sens] = list(set(sensitive))
    return newdict

def create_detailed_file(filename, dataset, sensitive_dict, tag):
    return results.ResultsFile(filename, dataset, sensitive_dict, tag)
    # f = open(filename, 'w')
    # f.write(get_detailed_metrics_header(dataset, sensitive_dict, tag) + '\n')
    # return f

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
