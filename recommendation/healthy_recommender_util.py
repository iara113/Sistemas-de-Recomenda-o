import cornac
import pandas as pd
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.constants import SEED
from recommenders.utils.timer import Timer
from tqdm.notebook import tqdm
from .. import utils
import pickle


def load_and_split():
    output_path = './data/cornac/'
    utils.ensure_dir(output_path)
    pp_interactions = pd.read_csv(output_path + 'foodData.csv', sep=',')
    pp_recipes = pd.read_csv(output_path + 'foodRecipes.csv', sep=',')
    train, test = python_random_split(pp_interactions, 0.8)
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

    # Load dicts
    with open(output_path + 'who_dict.pkl', 'rb') as file:
        who_dict = pickle.load(file)
    with open(output_path + 'fsa_dict.pkl', 'rb') as file:
        fsa_dict = pickle.load(file)
    with open(output_path + 'nutri_dict.pkl', 'rb') as file:
        nutri_dict = pickle.load(file)

    return pp_interactions, pp_recipes, train, test, train_set, who_dict, fsa_dict, nutri_dict


def train_multiple(models, train_set):
    with Timer() as t:
        for model in tqdm(models, total=len(models)):
            model.fit(train_set)
    print("It took {} minutes for training.".format(t.interval / 60))


def predict(model, train, store_results=True, i=0):
    all_predictions = predict_ranking(model, train, usercol='userID', itemcol='itemID', remove_seen=True)

    if store_results:
        output_path = './data/cornac/'
        utils.ensure_dir(output_path)
        model_key = str(i) + '_' + model.__class__.__name__
        path = output_path + model_key + '_interactions.csv'
        pd.DataFrame(all_predictions).to_csv(path, sep=',', index=False)

    return all_predictions


def predict_multiple(models, train, store_results=True):
    with Timer() as t:
        model_predictions = {}

        for index, model in tqdm(enumerate(models), total=len(models)):
            model_key = str(index) + '_' + model.__class__.__name__
            prediction = predict(model, train, store_results=store_results, i=index)
            model_predictions[model_key] = (model, {}, prediction)

    print("It took {} minutes for prediction of the models.".format(t.interval / 60))
    return model_predictions


def calc_mean_user_food_scores(recipe_to_who, recipe_to_fsa, recipe_to_nutri, predictions, k):
    """
    Calculate the mean WHO, FSA and NUTRI scores for each user.
    :param recipe_to_who: A dict with recipeID as key and WHO score as value. The range is 0-14, with 14 as the best.
    :param recipe_to_fsa: A dict with recipeID as key and FSA score as value. The range is 0-8, with 8 as the best.
    :param recipe_to_nutri: A dict with recipeID as key and NUTRI score as value. The range is A-E, with A as the best.
    :param predictions: The predictions of the model.
    :param k: The number of top predictions to consider.
    :return: The mean WHO, FSA and NUTRI scores for each user.
        The range of the who score is 0-14, with 14 as the best.
        The range of the fsa score is 0-8, with 8 as the best.
        The range of the mapped nutri score is 0-4, with 4 as the best.
    """
    who_user_scores = {}
    fsa_user_scores = {}
    nutri_user_scores = {}

    # Define the dict for the NUTRI score
    nutri_to_int_dict = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

    # Get top k predictions for each user
    top_k_predictions = predictions.groupby('userID').apply(lambda grp: grp.nlargest(k, 'prediction'))

    # Get scores from top k predictions
    for userID in top_k_predictions.userID.unique():
        user_matrix = top_k_predictions[top_k_predictions.userID == userID]

        # WHO, FSA and Nutri scores
        who_user_scores[userID] = _mean_score(user_matrix, utils.IdDict(), recipe_to_who)
        fsa_user_scores[userID] = _mean_score(user_matrix, utils.IdDict(), recipe_to_fsa)
        nutri_user_scores[userID] = _mean_score(user_matrix, nutri_to_int_dict, recipe_to_nutri)

    return who_user_scores, fsa_user_scores, nutri_user_scores


def _mean_score(user_matrix, score_norm, recipe_to_score):
    scores = user_matrix.itemID.apply(lambda x: score_norm[recipe_to_score[x]]).dropna()
    avg_score = sum(scores) / len(scores)
    return avg_score


def calc_mean_scores(user_score_dicts):
    result_scores = []

    # Get mean scores
    for user_score_dict in user_score_dicts:
        length = len(user_score_dict)
        if length > 0:
            result_scores.append(sum(user_score_dict.values()) / length)
        else:
            result_scores.append(0)

    return result_scores


def calc_score(recipe_to_who, recipe_to_fsa, recipe_to_nutri, test, model, post_processing_config, predictions, k,
               model_name, normalize=False):
    eval_map = map_at_k(test, predictions, col_prediction='prediction', k=k)
    eval_ndcg = ndcg_at_k(test, predictions, col_prediction='prediction', k=k)
    eval_precision = precision_at_k(test, predictions, col_prediction='prediction', k=k)
    eval_recall = recall_at_k(test, predictions, col_prediction='prediction', k=k)

    # Show scores
    index_name = '@' + str(k)
    evaluation = pd.DataFrame({index_name: [model_name]})
    evaluation.index = evaluation[index_name]
    evaluation = evaluation.drop(index_name, axis=1)

    # Model params
    evaluation['k'] = model.k
    evaluation['max_iter'] = model.max_iter
    evaluation['learning_rate'] = model.learning_rate
    evaluation['lambda_reg'] = model.lambda_reg

    # Post processing params
    for key, value in post_processing_config.items():
        if key == 'score_config':
            score_name, recipe_to_score = value
            evaluation['optimized_score'] = score_name
        else:
            evaluation[key] = value

    # Evaluation mesures
    evaluation['map'] = [eval_map]
    evaluation['ndcg'] = [eval_ndcg]
    evaluation['precision'] = [eval_precision]
    evaluation['recall'] = [eval_recall]

    # Evaluate user food scores
    who_user_scores, fsa_user_scores, nutri_user_scores = calc_mean_user_food_scores(recipe_to_who, recipe_to_fsa,
                                                                                     recipe_to_nutri, predictions, k)

    # Evaluate mean food scores
    avg_who_score, avg_fsa_score, avg_nutri_score = calc_mean_scores(
        (who_user_scores, fsa_user_scores, nutri_user_scores))

    # Normalize scores
    if normalize:
        avg_who_score = avg_who_score / 14
        avg_fsa_score = avg_fsa_score / 8
        avg_nutri_score = avg_nutri_score / 4

    evaluation['avg_who_score'] = [avg_who_score]
    evaluation['avg_fsa_score'] = [avg_fsa_score]
    evaluation['avg_nutri_score'] = [avg_nutri_score]

    return evaluation


def calc_scores(recipe_to_who, recipe_to_fsa, recipe_to_nutri, test, model_predictions, k, store_results=True,
                normalize=False, file_prefix=''):
    with Timer() as t:
        evaluations = {}
        for model_prediction in tqdm(model_predictions.items(), total=len(model_predictions)):
            name, (model, post_processing_config, predictions) = model_prediction
            evaluations[name] = calc_score(recipe_to_who, recipe_to_fsa, recipe_to_nutri, test, model,
                                           post_processing_config, predictions, k, name, normalize=normalize)

        evaluation_results = list(evaluations.values())
        result = pd.concat(evaluation_results, axis=0)

        if store_results:
            output_path = './data/cornac/'
            utils.ensure_dir(output_path)
            path = output_path + file_prefix + 'at_' + str(k) + '_results.csv'
            print('Writing file as ' + path)
            pd.DataFrame(result).to_csv(path, sep=',', index=True)

    print("It took {} minutes for evaluation.".format(t.interval / 60))
    return result
