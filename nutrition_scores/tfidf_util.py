import numpy as np
import pandas as pd
import score_util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommenders.utils.timer import Timer
from tqdm.notebook import tqdm
import utils


def calc_tfidf_similarities(pp_recipes):
    similarities = {}

    # Define the weights for each feature
    feature_weights = {
        'title': 0.3,
        'description': 0.1,
        'directions': 0.1,
        'tags': 0.2,
        'ingredients': 0.3
    }

    # Create a TF-IDF vectorizers for the recipes
    vectorizers = _vectorize_words(pp_recipes.copy())

    # build vocabulary
    for feature, matrix in vectorizers.items():
        similarities[feature] = cosine_similarity(vectorizers[feature])

    # Initialize an zero matrix for the overall similarities
    length = list(vectorizers.items())[0][1].shape[0]
    cosine_similarities = np.zeros((length, length))

    # Compute the overall similarity for each document
    for feature, matrix in vectorizers.items():
        feature_weight = feature_weights[feature]
        feature_similarities = similarities[feature]
        weighted_feature_similarities = feature_weight * feature_similarities
        cosine_similarities = cosine_similarities + weighted_feature_similarities

    return cosine_similarities


def _vectorize_words(pp_recipes):
    features = ['title', 'description', 'directions', 'tags', 'ingredients']
    vectorizers = {}

    for feature in features:
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

        # Replace Nan with empty string for current feature
        pp_recipes[feature].fillna('', inplace=True)

        tfidf_matrix = tf.fit_transform(pp_recipes[feature])
        vectorizers[feature] = tfidf_matrix

    return vectorizers


def get_healthy_substitutions(config_grid, cosine_similarities):
    with Timer() as t:
        healthy_substitution_pairs = []

        for config in tqdm(config_grid, desc="Substitution per configs"):
            # Unpack current model & post-processing parameters
            (score_name, recipe_to_score), substitution_threshold \
                = config['score_config'], config['substitution_threshold']

            # Define the normalization dicts for the scores
            if score_name == score_util.NUTRI_SCORE:
                score_norm = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
            else:
                score_norm = utils.IdDict()

            # Get "best" (healthiest and best matching) cosine similarities for each recipe
            healthy_substitutions = {}
            for recipeID in tqdm(range(cosine_similarities.shape[0]), leave=False, desc="Similarities per recipe"):
                # Default case: Recipe is not substituted
                healthy_substitutions[recipeID] = recipeID

                # Get the most similar recipes
                recipe_similarities = pd.DataFrame(cosine_similarities[recipeID]).sort_values(by=0, ascending=False)
                recipe_similarities = recipe_similarities.drop(recipeID, axis=0)  # Remove the recipe itself

                # Search a healthier recipe while being in bounds of the substitution similarity threshold
                for substitution_recipe in recipe_similarities.index:
                    if recipe_similarities.loc[substitution_recipe][0] >= substitution_threshold:
                        if score_norm[recipe_to_score[substitution_recipe]] > score_norm[recipe_to_score[recipeID]]:
                            healthy_substitutions[recipeID] = substitution_recipe
                            break
                    else:  # as recipes are sorted descending, we can break as soon as the similarity is too low
                        break

            # Add result
            healthy_substitution_pairs.append((healthy_substitutions, config))

    print("It took {} minutes for processing the healthy substitutions.".format(t.interval / 60))
    return healthy_substitution_pairs
