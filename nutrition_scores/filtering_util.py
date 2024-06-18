import score_util
from recommenders.utils.timer import Timer
from tqdm.notebook import tqdm
import pandas as pd
import utils


def multi_filter_recipes_by_threshold(config_grid, models_predictions, normalized_threshold):
    with Timer() as t:
        new_predictions = {}
        for model_prediction in tqdm(models_predictions.items(), total=len(models_predictions), desc="Models"):
            for config in tqdm(config_grid, leave=False, desc="Post-processing Configs"):
                # Unpack current model & post-processing parameters
                name, (model, _, predictions) = model_prediction
                (score_name, recipe_to_score), threshold = config['score_config'], config['threshold']

                # Filter predictions
                new_prediction = filter_recipes_by_threshold(score_name, recipe_to_score, predictions, threshold,
                                                             normalized_threshold)
                new_predictions[name + '+' + score_name + '+T=' + str(threshold)] = (model, config, new_prediction)

    print("It took {} minutes for filtering by threshold.".format(t.interval / 60))
    return new_predictions


def filter_recipes_by_threshold(score_name, recipe_to_score, model_predictions, threshold, normalized_threshold):
    # De-normalize threshold
    if normalized_threshold:
        if score_name == score_util.WHO_SCORE:
            threshold = threshold * 14
        elif score_name == score_util.FSA_SCORE:
            threshold = threshold * 8
        elif score_name == score_util.NUTRI_SCORE:
            threshold = threshold * 4

    # Define the normalization dicts for the scores
    if score_name == score_util.NUTRI_SCORE:
        score_norm = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    else:
        score_norm = utils.IdDict()

    # Filter: Create & apply mask
    mask = model_predictions.itemID.apply(lambda y: score_norm[recipe_to_score[y]] >= threshold)
    new_predictions = model_predictions[mask]

    return new_predictions


def multi_filter_recipes_by_percentage(config_grid, models_predictions):
    with Timer() as t:
        new_predictions = {}
        for model_prediction in tqdm(models_predictions.items(), total=len(models_predictions), desc="Models"):
            for config in tqdm(config_grid, leave=False, desc="Post-processing Configs"):
                # Unpack current model & post-processing parameters
                name, (model, _, predictions) = model_prediction
                (score_name, recipe_to_score), percentage = config['score_config'], config['percentage']

                # Filter predictions
                new_prediction = filter_recipes_by_percentage(score_name, recipe_to_score, predictions, percentage)
                new_predictions[name + '+' + score_name + '+P=' + str(percentage)] = (model, config, new_prediction)

    print("It took {} minutes for filtering by percentage.".format(t.interval / 60))
    return new_predictions


def filter_recipes_by_percentage(score_name, recipe_to_score, model_predictions, percentage):
    new_predictions = pd.DataFrame()

    # Define the normalization dicts for the scores
    if score_name == score_util.NUTRI_SCORE:
        score_norm = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    else:
        score_norm = utils.IdDict()

    # Rank all user recipes by the healthiness. keep the best percentage % recipes for each user.
    for user in model_predictions.userID.unique():
        user_recipes = model_predictions[model_predictions.userID == user]
        user_recipes['score'] = user_recipes.itemID.apply(lambda y: score_norm[recipe_to_score[y]])
        user_recipes = user_recipes.sort_values(by="score", ascending=False)
        user_recipes = user_recipes.head(int(len(user_recipes) * percentage))
        new_predictions = pd.concat([new_predictions, user_recipes])

    new_predictions = new_predictions.drop('score', axis=1)
    return new_predictions


def multi_filter_recipes_by_percentage_and_threshold(config_grid, models_predictions, normalized_threshold):
    with Timer() as t:
        new_predictions = {}
        for model_prediction in tqdm(models_predictions.items(), total=len(models_predictions), desc="Models"):
            for config in tqdm(config_grid, leave=False, desc="Post-processing Configs"):
                # Unpack current model & post-processing parameters
                name, (model, _, predictions) = model_prediction
                (score_name, recipe_to_score), percentage = config['score_config'], config['percentage']
                threshold = config['threshold']

                # Filter predictions
                new_prediction = filter_recipes_by_percentage_and_threshold(score_name, recipe_to_score, predictions,
                                                                            percentage, threshold, normalized_threshold)
                new_predictions[name + '+' + score_name + '+P=' + str(percentage) + '+T=' + str(threshold)] = \
                    (model, config, new_prediction)

    print("It took {} minutes for filtering by percentage and threshold.".format(t.interval / 60))
    return new_predictions


def filter_recipes_by_percentage_and_threshold(score_name, recipe_to_score, model_predictions, percentage, threshold,
                                               normalized_threshold):
    new_predictions = pd.DataFrame()

    # De-normalize threshold
    if normalized_threshold:
        if score_name == score_util.WHO_SCORE:
            threshold = threshold * 14
        elif score_name == score_util.FSA_SCORE:
            threshold = threshold * 8
        elif score_name == score_util.NUTRI_SCORE:
            threshold = threshold * 4

    # Define the normalization dicts for the scores
    if score_name == score_util.NUTRI_SCORE:
        score_norm = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    else:
        score_norm = utils.IdDict()

    # Rank all user recipes by the healthiness. keep the best percentage % recipes for each user.
    for user in model_predictions.userID.unique():
        user_recipes = model_predictions[model_predictions.userID == user]
        user_recipes['score'] = user_recipes.itemID.apply(lambda y: score_norm[recipe_to_score[y]])
        user_recipes = user_recipes.sort_values(by="score", ascending=False)
        user_recipes = user_recipes.head(int(len(user_recipes) * percentage))
        new_predictions = pd.concat([new_predictions, user_recipes])

    # Filter: Create & apply mask
    mask = model_predictions.itemID.apply(lambda y: score_norm[recipe_to_score[y]] >= threshold)
    new_predictions = new_predictions[mask]

    new_predictions = new_predictions.drop('score', axis=1)
    return new_predictions


def multi_exchange_recipes(models_predictions, healthy_substitution_pairs, k):
    with Timer() as t:
        new_predictions = {}
        for model_prediction in tqdm(models_predictions.items(), total=len(models_predictions), desc="Models"):
            for pair in tqdm(healthy_substitution_pairs, leave=False, desc="Post-processing Configs"):
                # Unpack current model & post-processing parameters
                name, (model, _, predictions) = model_prediction
                substitutions, config = pair
                (score_name, recipe_to_score), substitution_threshold \
                    = config['score_config'], config['substitution_threshold']

                # Exchange recipes
                new_prediction = exchange_recipes(predictions, substitutions, k)
                new_predictions[name + '+' + score_name + '+P=' + str(substitution_threshold)] \
                    = (model, config, new_prediction)

    print("It took {} minutes for recipe substitution.".format(t.interval / 60))
    return new_predictions


def exchange_recipes(model_predictions, healthy_substitutions, k):
    new_predictions = pd.DataFrame()

    # For each user substitute the k best recipes with healthier ones if possible
    for user in model_predictions.userID.unique():
        # Get recipes for user
        user_recipes = model_predictions[model_predictions.userID == user]

        # Get top-k recipes
        user_recipes = user_recipes.sort_values(by="prediction", ascending=False)
        user_recipes = user_recipes.head(k)

        # Get best cosine similarities and replace old recipes
        user_recipes['itemID'] = user_recipes.itemID.apply(lambda y: healthy_substitutions[y])

        # Append result
        new_predictions = pd.concat([new_predictions, user_recipes])

    return new_predictions
