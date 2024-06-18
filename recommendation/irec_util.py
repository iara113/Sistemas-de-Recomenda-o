# Import Metrics
import multiprocessing as mp
from os import listdir
from os.path import isfile, join
import ast

import numpy as np
import pandas as pd
from irec.offline_experiments.metrics.epc import EPC
from irec.offline_experiments.metrics.hits import Hits
from irec.offline_experiments.metrics.precision import Precision
from irec.offline_experiments.metrics.recall import Recall
from irec.offline_experiments.metrics.users_coverage import UsersCoverage


class Runner:
    train_dataset = []
    test_dataset = []
    eval_policy = None
    repetitions = 1
    processes = 8

    def __init__(self, train_dataset, test_dataset, eval_policy, repetitions=1, processes=8):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_policy = eval_policy
        self.repetitions = repetitions
        self.processes = processes

    class Looper:
        train_dataset = []
        test_dataset = []
        eval_policy = None
        agent = None

        def __init__(self, eval_policy, agent, train_dataset, test_dataset):
            self.eval_policy = eval_policy
            self.agent = agent
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset

        def train_single_agent(self):
            interactions, action_info = self.eval_policy.evaluate(self.agent, self.train_dataset, self.test_dataset)
            return interactions, action_info

    def _train_async(self, looper):
        # seems to produce same results - maybe because of async the random values are the same ones....
        async_results = []
        with mp.Pool(processes=self.processes) as pool:
            for _ in range(self.repetitions):
                async_results.append(pool.apply_async(looper.train_single_agent).get())
            pool.close()
            pool.join()

        return async_results

    def _train_sync(self, looper):
        sync_results = []
        for _ in range(self.repetitions):
            sync_results.append(looper.train_single_agent())

        return sync_results

    def train_multiple_agents(self, agents):
        result_map = {}
        i = 0  # to count methods- if a methods is executed under the same name twice

        for agent in agents:
            method_name = agent.name
            print('Starting with agent: ' + method_name + ' (' + str(i + 1) + '/' + str(len(agents)) + ')')

            looper = self.Looper(self.eval_policy, agent, self.train_dataset, self.test_dataset)

            # results = self._train_async(looper) #Disabled async training as it makes problems....
            results = self._train_sync(looper)

            result_map[(i, method_name)] = results
            path = './output/' + str(i) + '_' + method_name + '_interactions.csv'
            print('Writing file as ' + path)
            pd.DataFrame(results).to_csv(path, sep=',', index=False)
            i += 1

        return result_map


def load_interactions():
    result_map = {}
    directory = './output/'
    all_file_names = [f for f in listdir(directory) if isfile(join(directory, f))]

    for file_name in all_file_names:
        df_interactions = pd.read_csv(directory + file_name)
        model_index = file_name.split('_')[0]
        model_name = file_name.split('_')[1]
        results = df_interactions.values.tolist()
        for result in results:
            string_interaction = result[0]
            string_information = result[1]
            result[0] = ast.literal_eval(string_interaction)
            result[1] = ast.literal_eval(string_information)
        result_map[(model_index, model_name)] = results

    return result_map


def calc_scores(evaluator, interactions):
    # Getting the results
    hits_values = evaluator.evaluate(metric_class=Hits, results=interactions)
    precision_values = evaluator.evaluate(metric_class=Precision, results=interactions)
    recall_values = evaluator.evaluate(metric_class=Recall, results=interactions)
    epc_values = evaluator.evaluate(metric_class=EPC, results=interactions)
    usr_cov_values = evaluator.evaluate(metric_class=UsersCoverage, results=interactions)

    # Show scores
    evaluation = pd.DataFrame({'': ['@' + str(x) for x in evaluator.interactions_to_evaluate]})
    evaluation.index = evaluation['']
    evaluation = evaluation.drop('', axis=1)
    evaluation['hits'] = [np.mean(list(x.values())) for x in hits_values]
    evaluation['precision'] = [np.mean(list(x.values())) for x in precision_values]
    evaluation['recall'] = [np.mean(list(x.values())) for x in recall_values]
    evaluation['epc'] = [np.mean(list(x.values())) for x in epc_values]
    evaluation['usr_cov'] = [np.mean(list(x.values())) for x in usr_cov_values]

    return evaluation


def calc_avg_scores(evaluator, interaction_set):
    results = []
    for interactions, _ in interaction_set:
        results.append(calc_scores(evaluator, interactions))

    # Build average dataset
    evaluation = pd.DataFrame({'': ['@' + str(x) for x in evaluator.interactions_to_evaluate]})
    evaluation.index = evaluation['']
    evaluation = evaluation.drop('', axis=1)
    evaluation['hits'] = pd.concat([result[['hits']].T for result in results]).mean(axis=0)
    evaluation['precision'] = pd.concat([result[['precision']].T for result in results]).mean(axis=0)
    evaluation['recall'] = pd.concat([result[['recall']].T for result in results]).mean(axis=0)
    evaluation['epc'] = pd.concat([result[['epc']].T for result in results]).mean(axis=0)
    evaluation['usr_cov'] = pd.concat([result[['usr_cov']].T for result in results]).mean(axis=0)

    return evaluation


def calc_multiple_scores(evaluator, results):
    evaluations = {id_name: calc_avg_scores(evaluator, interaction_set) for id_name, interaction_set in results.items()}
    method_names = [name for id, name in evaluations.keys()]
    evaluation_results = list(evaluations.values())
    result = []

    for i in range(0, len(evaluation_results[0])):  # for each '@'-value
        # add results
        i_th_rows = [evaluation.iloc[[i]] for evaluation in evaluation_results]
        merged = pd.concat(i_th_rows, axis=0)

        # add method names as y-axis
        table_name = str(i_th_rows[0].index[0])
        merged[table_name] = [x for x in method_names]
        merged.index = merged[table_name]
        merged = merged.drop(table_name, axis=1)

        result.append(merged)

    return result
