import cornac
import pandas as pd
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.constants import SEED
from recommenders.utils.timer import Timer

import utils


def load_and_split():
    output_path = './data/cornac/'
    utils.ensure_dir(output_path)
    pp_interactions = pd.read_csv(output_path + 'foodData.csv', sep=',')
    train, test = python_random_split(pp_interactions, 0.75)
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)
    return pp_interactions, train, test, train_set


def train(model, train_set):
    with Timer() as t:
        model.fit(train_set)
    print(model.__class__.__name__ + ": Took {} seconds for training.".format(t))


def train_multiple(models, train_set):
    for model in models:
        train(model, train_set)


def predict(model, train, store_results=True, i=0):
    with Timer() as t:
        all_predictions = predict_ranking(model, train, usercol='userID', itemcol='itemID', remove_seen=True)
    print(model.__class__.__name__ + ": Took {} minutes for prediction of model.".format(t.interval / 60))

    if store_results:
        output_path = './data/cornac/'
        utils.ensure_dir(output_path)
        model_key = str(i) + '_' + model.__class__.__name__
        path = output_path + model_key + '_interactions.csv'
        print('Writing file as ' + path)
        pd.DataFrame(all_predictions).to_csv(path, sep=',', index=False)

    return all_predictions


def predict_multiple(models, train, store_results=True):
    model_predictions = {}

    for index, model in enumerate(models):
        model_key = str(index) + '_' + model.__class__.__name__
        prediction = predict(model, train, store_results=store_results, i=index)
        model_predictions[model_key] = prediction

    return model_predictions


def calc_score(test, predictions, k, model_name):
    with Timer() as t:
        print("Start map evaluation.")
        eval_map = map_at_k(test, predictions, col_prediction='prediction', k=k)
        print("Start ndcg evaluation.")
        eval_ndcg = ndcg_at_k(test, predictions, col_prediction='prediction', k=k)
        print("Start precision evaluation.")
        eval_precision = precision_at_k(test, predictions, col_prediction='prediction', k=k)
        print("Start recall evaluation.")
        eval_recall = recall_at_k(test, predictions, col_prediction='prediction', k=k)

        # Show scores
        index_name = '@' + str(k)
        evaluation = pd.DataFrame({index_name: [model_name]})
        evaluation.index = evaluation[index_name]
        evaluation = evaluation.drop(index_name, axis=1)
        evaluation['map'] = [eval_map]
        evaluation['ndcg'] = [eval_ndcg]
        evaluation['precision'] = [eval_precision]
        evaluation['recall'] = [eval_recall]
    print(model_name + index_name + ": Took {} minutes for evaluation.".format(t.interval / 60))

    return evaluation


def calc_scores(test, model_predictions, k_values, store_results=True):
    result = []
    for k in k_values:
        evaluations = {name: calc_score(test, predictions, k, name) for name, predictions in model_predictions.items()}
        evaluation_results = list(evaluations.values())
        merged = pd.concat(evaluation_results, axis=0)
        result.append(merged)

        if store_results:
            output_path = './data/cornac/'
            utils.ensure_dir(output_path)
            path = output_path + 'at_' + str(k) + '_results.csv'
            print('Writing file as ' + path)
            pd.DataFrame(merged).to_csv(path, sep=',', index=True)

    return result
