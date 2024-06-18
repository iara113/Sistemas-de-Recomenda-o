import ast
import gzip
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandarallel import pandarallel


class IdDict(dict):
    def __missing__(self, key):
        return key


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _try_parse(string_object):
    try:
        return json.loads(string_object)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(string_object)
        except SyntaxError:
            raise SyntaxError('String not parseable.')
        except ValueError:
            raise SyntaxError('String not parseable.')


def _parse_string_list(string_object):
    try:
        return _try_parse(string_object)
    except SyntaxError:
        try:
            return _try_parse(string_object.replace(',Serve.', ''))
        except SyntaxError:
            try:
                return [s.replace('"', "").replace('\'', "").replace('[', "").replace(']', "").strip() for s in
                        string_object.split('","')]
            except Exception:
                return None


def _parse_string_triple(string_object):
    try:
        return _try_parse(string_object)
    except SyntaxError:
        try:
            # try to parse fraction numbers
            try:
                temp_r = string_object.split('",', 1)[1]
            except IndexError:
                temp_r = string_object.split(',', 1)[1]
            try:
                temp_fraction = temp_r.split(',"', 1)[0]
                if len(temp_fraction) == 1:
                    temp_fraction = temp_r.split(',', 1)[0]
            except IndexError:
                temp_fraction = temp_r.split(',', 1)[0]
            numbers = temp_fraction.split('/')
            fraction = float(numbers[0]) / float(numbers[1])

            # rebuild string
            split_string_object = string_object.split(temp_fraction, 1)
            new_string_object = split_string_object[0] + str(fraction) + split_string_object[1]

            # parse with ast again....
            return ast.literal_eval(new_string_object)

        except Exception:
            splits = re.split('",|,"|\(,|,[0-9]|[0-9],', string_object)

            if len(splits) > 3:
                number_of_join = len(splits) - 2
                new_splits = []
                new_splits.append(splits[0])
                new_splits.append(splits[1])
                new_splits.append(''.join(splits[number_of_join:]))  # add the last n'th elements together
                splits = new_splits

            return [s.replace('"', "").replace('\'', "").replace('[', "")
                    .replace(']', "").replace('(', "").replace(')', "").strip() for s in splits]


def _try_split(split_words, name):
    for split_word in split_words:
        split_word = split_word + ' '
        try:
            quantity_part, name_part = name.split(split_word, 1)
            quantity_part = quantity_part + split_word
            return quantity_part, name_part
        except ValueError:
            pass

    # if nothing worked use default case
    return '', name


def _shift_quantity_information(ingredient_triple):
    category = ingredient_triple[0]
    quantity = ingredient_triple[1]
    name = ingredient_triple[2]

    # Build sequential list to split, start with entities like 'bag' to split, if this does not work use the units
    # and else ')' to split
    split_words = ['bag', 'can', 'cup', 'bags', 'cans', 'cups', 'ounces', 'ounce', 'pound', 'pounds', 'teaspoon',
                   'teaspoons', 'tablespoon', 'tablespoons', 'g', 'kg', 'lb', 'lbs', 'ml', 'oz', 'pint', 'tbsp', 'tsp',
                   ')']

    # Split quantity from name
    quantity_part, name_part = _try_split(split_words, name)

    # Add quantity part to the overall quantity
    quantity = str(quantity) + ' time(s) ' + quantity_part

    # Remove quantity part from name
    name = name_part.strip()

    return category, name, quantity


def _is_ingredient_in_key(ingredient, key):
    if len(ingredient) == 3:
        ingr_key = ingredient[0]

        if ingr_key == key:
            try:
                amount = float(ingredient[1])
            except ValueError:
                amount = ingredient[1]
            name = ingredient[2]
            return amount, name
        else:
            return None
    else:  # ingredient not valid, parse was not successful
        return None


def _get_ingredients_for_key(ingredients, key):
    keyed_ingredients = [_is_ingredient_in_key(ingredient, key) for ingredient in ingredients]
    keyed_ingredients = [x for x in keyed_ingredients if x is not None]

    return keyed_ingredients


def _count_parsed_list(parsed_list):
    if parsed_list is not None:
        return int(len(parsed_list))
    else:
        return -1


def _count_ingredient_list(ingredient_dict, parse_as_dict):
    if ingredient_dict is not None:
        if parse_as_dict:
            return sum([len(sub_ingredients) for sub_ingredients in ingredient_dict.values()])
        else:
            return len(ingredient_dict)
    else:
        return -1


def _parse_layered_ingredients(string_object, parse_as_dict=True):
    try:
        string_ingredients = json.loads(string_object)
        parsed_ingredients = [_parse_string_triple(ingredient) for ingredient in string_ingredients]

        # shift quantity information from ingredient name to quantity field
        parsed_ingredients = [_shift_quantity_information(ingredient_triple) for ingredient_triple in
                              parsed_ingredients]

        if parse_as_dict:
            # get all keys, set comprehension to ensure uniqueness, also do not include invalid keys
            keys = {ingredient[0] for ingredient in parsed_ingredients if isinstance(ingredient[0], str)}

            # add ingredients to keys
            ingredient_dict = {key: _get_ingredients_for_key(parsed_ingredients, key) for key in keys}

            return ingredient_dict
        else:
            ingredient_list = [ingredient[1] for ingredient in parsed_ingredients]
            return ingredient_list
    except json.JSONDecodeError:
        return None


def _lookup_index(row, dict):
    try:
        return dict[row[0]]
    except KeyError:
        return -1


def _row_is_in_tuple(row, item_user_set):
    is_in = (row[0], row[1]) in item_user_set
    return is_in


def food_locator_to_food_com(food_locator_url):
    return food_locator_url.replace('http://idea.rpi.edu/heals/kb/recipe/', 'https://www.food.com/recipe/')


def load_and_clean_data(data_location, additional_location, s3=None, add_recipe_columns=None,
                        add_review_columns=None, comment_relations=2.5, authorship_relations=-1, recipe_tags=False,
                        k_user=2, k_recipe=2, debug=False, exclude_author_ids=True, keep_old_ids=False,
                        parse_as_dict=True):
    """
    Loads and cleans the dataset from food.com (including possibly some information from other sources)

    1. [Optional] Either removes comments (== ratings of value 0) or modifies those to a certain rating value.
    2. [Optional] Adds authorship relations (from the recipe data) as rating relations.
    3. [Optional] Adds tags for recipes from another food.com dataset (https://www.aclweb.org/anthology/D19-1613/)
    4. [Optional] Uses threshold discarding with specified minimum of user and recipe interactions.
    5. Reset indices of members and recipes to be in a continuous space starting by 0.
    6. Parse average recipe rating and amount
    7. Parses nutrients (multiple columns in grams and milligrams respectively)
    8. Cleans durations dates (minutes) into easy-to-access formats.
    9. Counts the directions sizes for each recipe.
    10. Counts the ingredients sizes for each recipe.

    :param data_location: Name of the directory where the data is located in.
    :param additional_location: Name of the directory where the additional food.com dataset is located in
        (https://www.aclweb.org/anthology/D19-1613/).
    :param s3: The s3 connection. When the connection is not None, tries to use this connection for data loading instead
        local data loading.
    :param add_recipe_columns: List of additional/non-standard columns/features which shall be present in the
        recipe dataset. None or [] if not necessary.
    :param add_review_columns: List of additional/non-standard columns/features which shall be present in the
        review dataset. None or [] if not necessary.
    :param comment_relations: Defines whether the comment relations shall be added:
        -1 --> not added,
        any other number >= 0 --> add those relations with the given rating.
    :param authorship_relations: Defines whether the authorship relations shall be added:
        -1 --> not added,
        any other number >= 0 --> add those relations with the given rating.
    :param recipe_tags: When recipe_tags are true, add a tags column from another food.com dataset for most recipes,
        skips otherwise.
    :param k_recipe: The required minimum recipe interactions.
    :param k_user: The required minimum user interactions.
    :param debug: If debug messages shall be printed.
    :param exclude_author_ids: Whether the author ids shall also remapped. When True and given authors did not write
        reviews (or excluded via threshhold discading), then the author ids will be invalid.
    :param keep_old_ids: Whether old ids should be kept or discarded.
    :param parse_as_dict: Whether ingredients shall be parsed as a dict including additional information instead of a
        simple list of ingredients.
    :return:
        recipes, reviews, users, recipes_dict_inv (new_id --> old_id), member_dict_inv (new_id --> old_id),
        food_locator_dict (old_id --> food_kg_locator), food_com_dict (recipe_id --> recipe_url),
        data (merged recipes & reviews)
    """

    # compute column names of recipes & reviews/ratings
    recipe_column_names = ['recipe_id', 'title', 'description', 'author_id', 'duration', 'nutrition', 'directions',
                           'ingredients', 'serves', 'rating', 'last_changed_date', 'food_kg_locator', 'recipe_url']
    review_column_names = ['member_id', 'recipe_id', 'rating', 'text', 'likes', 'last_modified_date']
    member_column_names = ['member_id', 'member_url', 'member_name', 'member_description', 'member_status',
                           'member_avg_rating', 'member_map_pin', 'member_joined', 'follows_count', 'follow_me_count',
                           'avatar_url']
    if add_recipe_columns is not None:
        recipe_column_names.extend(add_recipe_columns)
    if add_review_columns is not None:
        review_column_names.extend(add_review_columns)

    # read recipes & reviews
    recipes, reviews, users = _read_recipes_and_reviews(data_location, recipe_column_names, review_column_names,
                                                        member_column_names, s3, debug)

    # Reparse invalid ratings/comments with value 0 to another value based on param
    reviews = _parse_comments(reviews, comment_relations, debug)

    # Add authorship relations as rating based on param
    reviews = _parse_authorship(recipes, reviews, authorship_relations, debug)

    # Add recipe tags from other dataset based on param
    _add_external_tags(additional_location, recipes, recipe_tags, s3, debug)

    # Threshold discarding
    recipes, reviews, users = _threshold_discarding_init(k_recipe, k_user, recipes, reviews, users, debug)

    # Generate dicts to fastly lookup the new indices
    member_dict, recipes, recipes_dict, reviews, users, food_locator_dict, food_com_dict = _reindex(recipes, reviews,
                                                                                                    users,
                                                                                                    exclude_author_ids,
                                                                                                    keep_old_ids, debug)

    # parse average recipe rating and amount
    _parse_recipe_rating_and_amount(recipes, debug)

    # parse nutrition elements
    _parse_nutrients(recipes, debug)

    # Clean durations
    _parse_durations(recipes, debug)

    # Count Directions and Ingredients
    _count_directions_and_ingredients(recipes, debug, parse_as_dict)

    # invert dictionaries (now: new_id --> old_id)
    if debug:
        print("[11] Finalize dictionaries ...")
    recipes_dict_inv = {v: k for k, v in recipes_dict.items()}
    member_dict_inv = {v: k for k, v in member_dict.items()}

    # Merge data
    data = pd.merge(recipes, reviews, right_on='new_recipe_id', left_on='new_recipe_id')

    return recipes, reviews, users, recipes_dict_inv, member_dict_inv, food_locator_dict, food_com_dict, data


def _read_recipes_and_reviews(data_location, recipe_column_names, review_column_names, member_column_names, s3, debug):
    parser = '%d/%m/%Y'
    user_parser = '%m/%Y'

    if debug:
        print("[0] Read recipes, reviews & users ...")
    if s3 is None:
        recipes = pd.read_csv(data_location + 'recipes.csv', sep=',', parse_dates=['last_changed_date'])
        reviews = pd.read_csv(data_location + 'reviews.csv', sep=',', parse_dates=['last_modified_date'])
        users = pd.read_csv(data_location + 'members.csv', sep=',', parse_dates=['member_joined'])

        #recipes = pd.read_csv(data_location + 'recipes.csv', sep=',', parse_dates=['last_changed_date'],
          #                    date_format=parser).loc[:, recipe_column_names]
        #reviews = pd.read_csv(data_location + 'reviews.csv', sep=',', parse_dates=['last_modified_date'],
         #                     date_format=parser).loc[:, review_column_names]
        #users = pd.read_csv(data_location + 'members.csv', sep=',', parse_dates=['member_joined'],
         #                   date_format=user_parser).loc[:, member_column_names]
    else:
        with s3.open(data_location + 'recipes.csv') as file:
            recipes = pd.read_csv(file, sep=',', parse_dates=['last_changed_date'], date_format=parser).loc[:,
                      recipe_column_names]
        with s3.open(data_location + 'reviews.csv') as file:
            reviews = pd.read_csv(file, sep=',', parse_dates=['last_modified_date'], date_format=parser).loc[:,
                      review_column_names]
        with s3.open(data_location + 'members.csv') as file:
            users = pd.read_csv(file, sep=',', parse_dates=['member_joined'], date_format=user_parser).loc[:,
                    member_column_names]
    return recipes, reviews, users


def _parse_comments(reviews, comment_relations, debug):
    if comment_relations == -1:  # remove the 0 ratings (==comments)
        if debug:
            print("[1] Remove comment relations ...")
        reviews = reviews[reviews.rating != 0]
    if comment_relations >= 0:  # set the new value for the 0 ratings
        if debug:
            print("[1] Reparse comment relations ...")
        reviews.loc[reviews.rating == 0, 'rating'] = comment_relations
    return reviews


def _parse_authorship(recipes, reviews, authorship_relations, debug):
    if authorship_relations >= 0:
        if debug:
            print("[2] Add authorship relations ...")
        # compute new potential ratings and rename columns such that they are similar to the reviews
        relations = recipes.loc[:, ['author_id', 'recipe_id', 'last_changed_date']]
        relations['rating'] = authorship_relations
        relations['text'] = 'NaN'
        relations = relations.rename(columns={'author_id': 'member_id', 'last_changed_date': 'last_modified_date'})

        # get a matrix of already existing item-user ratings, calc valid indices, remove invalid ones from relation
        review_set = set(zip(reviews.recipe_id, reviews.member_id))
        relation_set = set(map(tuple, relations[['recipe_id', 'member_id']].to_numpy().tolist()))
        intersection = relation_set.intersection(review_set)
        valid_indices = [not _row_is_in_tuple(row, intersection) for row in
                         relations[['recipe_id', 'member_id']].to_numpy().tolist()]
        relations = relations[valid_indices]

        # add relations to reviews
        reviews = pd.concat([reviews, relations], ignore_index=True)
    else:
        if debug:
            print("[2] Skip authorship relations ...")
    return reviews


def _add_external_tags(additional_location, recipes, recipe_tags, s3, debug):
    if recipe_tags is True:
        if debug:
            print("[3] Add recipe tags ...")
        # Load additional data
        if s3 is None:
            additional_data = pd.read_csv(additional_location + 'RAW_recipes.csv', sep=',').loc[:,
                              ['id', 'name', 'tags']]
        else:
            with s3.open(additional_location + 'RAW_recipes.csv') as file:
                additional_data = pd.read_csv(file, sep=',').loc[:, ['id', 'name', 'tags']]

        # Add tags for same recipes
        recipes['tags'] = recipes['recipe_id'].map(additional_data.set_index('id')['tags'])
        recipes['tags'] = [_try_parse(row) if isinstance(row, str) else row for row in
                           recipes['tags'].to_numpy().tolist()]
    else:
        if debug:
            print("[3] Skip recipe tags ...")


def _threshold_discarding_init(k_recipe, k_user, recipes, reviews, users, debug):
    if (k_user > 1 and k_recipe > 1) or (k_user >= 1 and k_recipe > 1) or (k_user > 1 and k_recipe >= 1):
        if debug:
            print("[4] Start Threshold Discarding ...")
        recipes, reviews, users = threshold_discarding(k_user, k_recipe, recipes, reviews, users, debug)
    else:
        if debug:
            print("[4] Skip Threshold Discarding ...")
    return recipes, reviews, users


def _reindex(recipes, reviews, users, exclude_author_ids, keep_old_ids, debug):
    if debug:
        print("[5] Reset indices ...")

    # get the food_locator mapping: recipe_id --> food_kg_locator
    food_locator_dict = recipes.set_index('recipe_id').to_dict()['food_kg_locator']

    # get the food_com mapping: recipe_id --> recipe_url
    food_com_dict = recipes.set_index('recipe_id').to_dict()['recipe_url']

    # resets index twice to end up with continuous indices in the dictionaries
    recipes_dict = recipes.reset_index().reset_index().set_index('recipe_id').to_dict()['level_0']

    # resets index twice to end up with continuous indices in the dictionaries
    if exclude_author_ids:
        all_member_ids = reviews['member_id'].squeeze()
        single_member_ids = all_member_ids.drop_duplicates().reset_index().reset_index()
        member_dict = single_member_ids.set_index('member_id').to_dict()['level_0']
    else:  # also add author ids
        all_member_ids = pd.concat([recipes['author_id'].squeeze(), reviews['member_id'].squeeze()])
        single_member_ids = all_member_ids.drop_duplicates().reset_index().reset_index()
        member_dict = single_member_ids.set_index(0).to_dict()['level_0']

    # lookup the new indices and add them as a new column
    reviews['new_member_id'] = [_lookup_index(row, member_dict) for row in reviews[['member_id']].to_numpy().tolist()]
    users['new_member_id'] = [_lookup_index(row, member_dict) for row in users[['member_id']].to_numpy().tolist()]
    reviews['new_recipe_id'] = [_lookup_index(row, recipes_dict) for row in reviews[['recipe_id']].to_numpy().tolist()]
    recipes['new_recipe_id'] = [_lookup_index(row, recipes_dict) for row in recipes[['recipe_id']].to_numpy().tolist()]

    # Also update author_ids, invalid ids will be mapped with -1
    recipes['new_author_id'] = [_lookup_index(row, member_dict) for row in recipes[['author_id']].to_numpy().tolist()]

    # drop old indices
    if not keep_old_ids:
        reviews = reviews.drop(columns=['member_id', 'recipe_id'])
        recipes = recipes.drop(columns=['recipe_id', 'author_id'])
        users = users.drop(columns=['member_id'])
    return member_dict, recipes, recipes_dict, reviews, users, food_locator_dict, food_com_dict


def _parse_recipe_rating_and_amount(recipes, debug):
    if debug:
        print("[6] Parse average recipe rating and amount ...")
    json_data = recipes['rating']
    parsed_json = [ast.literal_eval(json_string) for json_string in json_data]
    recipes['average_rating'] = [x[0] for x in parsed_json]
    recipes['number_of_ratings'] = [x[1] for x in parsed_json]

    # remove old rating column
    recipes.drop(['rating'], axis=1, inplace=True)


def _parse_nutrients(recipes, debug):
    if debug:
        print("[7] Parse nutrients ...")
    json_data = recipes.loc[:, ['nutrition']]['nutrition']
    parsed_json = [json.loads(json_string) for json_string in json_data]
    nutrition_data = pd.json_normalize(parsed_json)

    # remove units from data
    nutrition_normalized = pd.DataFrame()
    nutrition_normalized['servingSize [g]'] = nutrition_data['servingSize'].map(
        lambda x: x.rsplit('(')[1].rsplit(')')[0])
    nutrition_normalized['servingsPerRecipe'] = nutrition_data['servingPerRecipe']
    nutrition_normalized['calories [cal]'] = nutrition_data['calories']
    nutrition_normalized['caloriesFromFat [cal]'] = nutrition_data['caloriesFromFat'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['totalFat [g]'] = nutrition_data['totalFat'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['saturatedFat [g]'] = nutrition_data['saturatedFat'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['cholesterol [mg]'] = nutrition_data['cholesterol'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['sodium [mg]'] = nutrition_data['sodium'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['totalCarbohydrate [g]'] = nutrition_data['totalCarbohydrate'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['dietaryFiber [g]'] = nutrition_data['dietaryFiber'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['sugars [g]'] = nutrition_data['sugars'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['protein [g]'] = nutrition_data['protein'].map(lambda x: x.rsplit(' ')[0])

    # add nutrition data to the recipes and convert to float/int
    recipes[['servingsPerRecipe']] = nutrition_normalized[['servingsPerRecipe']].astype(int)
    recipes[['servingSize [g]', 'calories [cal]', 'caloriesFromFat [cal]', 'totalFat [g]', 'saturatedFat [g]',
             'cholesterol [mg]', 'sodium [mg]', 'totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]',
             'protein [g]']] = \
        nutrition_normalized[
            ['servingSize [g]', 'calories [cal]', 'caloriesFromFat [cal]', 'totalFat [g]', 'saturatedFat [g]',
             'cholesterol [mg]', 'sodium [mg]', 'totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]',
             'protein [g]']].astype(float)

    # remove old nutrition column
    recipes.drop(['nutrition'], axis=1, inplace=True)


def _parse_durations(recipes, debug):
    if debug:
        print("[8] Clean durations ...")
    date_time = pd.to_datetime(recipes['duration'].replace({'min': ' min'}, regex=True), format='%M min',
                               errors="coerce").fillna(
        pd.to_datetime(recipes['duration'].replace({'mins': ' min'}, regex=True), format='%M min',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hr': ' hour'}, regex=True), format='%H hour',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hrs': ' hour'}, regex=True), format='%H hour',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hr': ' hour', 'min': ' min'}, regex=True), format='%H hour %M min',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hr': ' hour', 'mins': ' min'}, regex=True),
                       format='%H hour %M min',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hrs': ' hour', 'min': ' min'}, regex=True),
                       format='%H hour %M min',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hrs': ' hour', 'mins': ' min'}, regex=True),
                       format='%H hour %M min',
                       errors="coerce"))
    time = date_time.dt.hour * 60 + date_time.dt.minute
    recipes['duration'] = time


def _count_directions_and_ingredients(recipes, debug, parse_as_dict):
    # parse directions
    if debug:
        print("[9] Parse & Count directions ...")
    list_data_directions = recipes.loc[:, ['directions']]['directions'].map(
        lambda x: x.replace('{', '[').replace('}', ']').replace('\r', '').replace('\t', ''))
    list_data_directions = [_parse_string_list(direction) for direction in list_data_directions]
    direction_sizes = [_count_parsed_list(direction) for direction in list_data_directions]

    # parse ingredients
    if debug:
        print("[10] Parse & Count ingredients ...")
    list_data_ingredients = recipes.loc[:, ['ingredients']]['ingredients'].map(
        lambda x: x.replace('{', '[').replace('}', ']').replace('\r', '').replace('\t', ''))
    list_data_ingredients = [_parse_layered_ingredients(ingredient, parse_as_dict) for ingredient in
                             list_data_ingredients]
    ingredients_sizes = [_count_ingredient_list(ingredient, parse_as_dict) for ingredient in list_data_ingredients]

    # add both sizes and parsed lists to recipes
    recipes['directions'] = list_data_directions
    recipes['direction_size'] = direction_sizes
    recipes['ingredients'] = list_data_ingredients
    recipes['ingredients_sizes'] = ingredients_sizes


def threshold_discarding(k_user, k_recipe, recipes, reviews, users, debug):
    """
    Executes the threshold discarding on recipe and review data frames.

    :param k_user: Number of minimal user interactions. Must be at least 2.
    :param k_recipe: Number of minimal recipe interactions. Must be at least 2.
    :param recipes: Dataframe of the recipes.
    :param reviews: Dataframe of the reviews.
    :param users: Dataframe of the users.
    :param debug: If debug messages shall be printed.
    :return:
        reduced_recipes: The recipe data frame with recipes with have at least k_recipe interactions,
        reduced_reviews: The review data frame with reviews with have at least k_user interactions,
    """
    # Count occurrence of each user and recipe in reviews
    counts_col_user = reviews.groupby("member_id")["member_id"].transform(len)
    counts_col_recipes = reviews.groupby("recipe_id")["recipe_id"].transform(len)

    # Bool masks
    mask_user = counts_col_user >= k_user
    mask_recipes = counts_col_recipes >= k_recipe

    # Count reviews with non-sparse users and non-sparse recipes
    _count(k_user, mask_user, 'user')
    _count(k_recipe, mask_recipes, 'recipe')

    # Apply masks for reviews
    if debug:
        print(f"   Number of reviews before: {reviews.shape[0]}")
    reduced_reviews = reviews[mask_user & mask_recipes]
    if debug:
        print(f"   Number of reviews after: {reduced_reviews.shape[0]}")

    # Apply masks for recipes
    if debug:
        print(f"   Number of recipes before: {recipes.shape[0]}")
    reduced_recipes = recipes[recipes.recipe_id.isin(reduced_reviews.recipe_id.unique())]
    if debug:
        print(f"   Number of recipes after: {reduced_recipes.shape[0]}")

    # Apply masks for users
    if debug:
        print(f"   Number of users before: {users.shape[0]}")
    reduced_users = users[users.member_id.isin(reduced_reviews.member_id.unique())]
    if debug:
        print(f"   Number of users after: {reduced_users.shape[0]}")

    # resets indices
    reduced_reviews = reduced_reviews.reset_index()
    reduced_reviews.drop(['index'], axis=1, inplace=True)
    reduced_recipes = reduced_recipes.reset_index()
    reduced_recipes.drop(['index'], axis=1, inplace=True)
    reduced_users = reduced_users.reset_index()
    reduced_users.drop(['index'], axis=1, inplace=True)

    return reduced_recipes, reduced_reviews, reduced_users


def _count(k, mask, name):
    counts = mask.value_counts()
    try:
        counts_true = counts[True]
    except KeyError:
        counts_true = 0
    try:
        counts_false = counts[False]
    except KeyError:
        counts_false = 0
    print("   Data with non-sparse " + name + " (k_" + name + ": " + str(k) + "):")
    print(f"       #True: {int(counts_true / len(mask) * 100)}%")
    print(f"       #False: {int(counts_false / len(mask) * 100)}%")


def get_recipe_ingredients_dict(recipes_df, ingredients_df):
    """
    Returns a dictionary with the recipe urls as keys and the ingredients as values.

    :param recipes_df: The recipe dataframe.
    :param ingredients_df: The ingredients dataframe.
    :return: The dictionary with the recipe urls as keys and the ingredients as values.
    """
    # get the recipe ids
    recipe_ids = recipes_df['new_recipe_id'].to_numpy().tolist()

    # get the recipe urls
    recipe_url_dict = recipes_df.set_index('new_recipe_id')['recipe_url'].to_dict()

    # get the ingredients
    recipe_ingredient_dict = ingredients_df.groupby('recipe_id')['ingredient_id'].agg(list).to_dict()

    # build the dictionary
    recipe_ingredients_dict = {recipe_url_dict[recipe_id]: recipe_ingredient_dict.get(recipe_id, []) for recipe_id in
                               recipe_ids}

    return recipe_ingredients_dict


def get_ingredient_names_dict(ingredients_df):
    """
    Returns a dictionary with the ingredient ids as keys and the ingredient names as values.

    :param ingredients_df: The ingredients dataframe.
    :return: The dictionary with the ingredient ids as keys and the ingredient names as values.
    """
    # get the ingredient names
    ingredient_name_dict = ingredients_df.set_index('ingredient_id')['ing_name'].to_dict()

    return ingredient_name_dict


def load_ingredient_dict(recipes, graph_location, s3=None):
    """
    Loads ingredient specific data.

    :param recipes: Those recipe define which ingredients shall be fetched- only connected ingredients will be included.
    :param graph_location: Name of the directory where the graph related data is located in.
    :param s3: The s3 connection. When the connection is not None, tries to use this connection for data loading instead
        local data loading.
    :return:
        reduced_ingredients: A dataframe with following features: ing_name, recipe_id, ingredient_id
        ingredients_dict: A dictionary for the ingredient ids (ingredient_id -> foodKG_mapped_ingredient)
        mapped_recipes: A list of all mapped recipes (might be shorter than the size of param recipes as not mappings
            might be incomplete)
    """
    # read ingredient data
    if s3 is None:
        ingredients = pd.read_csv(graph_location + 'graph_ingredient_mapping.csv', sep=',').loc[:,
                      ['ing_name', 'recipe', 'foodKG_mapped_ingredient']]
    else:
        with s3.open(graph_location + 'graph_ingredient_mapping.csv') as file:
            ingredients = pd.read_csv(file, sep=',').loc[:, ['ing_name', 'recipe', 'foodKG_mapped_ingredient']]

    # build food_kg_locator_dict such as food_kg_locator --> new_recipe_id
    # Note: As the recipes are the crawled ones, there might be the case that they don't exist (because they were not
    # present on food.com anymore)
    food_kg_locator_dict = recipes.reset_index().set_index('food_kg_locator').to_dict()['new_recipe_id']

    # use recipes_dict (food_kg_locator --> new_recipe_id)
    ingredients['recipe_id'] = [_lookup_index(row, food_kg_locator_dict) for row in
                                ingredients[['recipe']].to_numpy().tolist()]

    # Get the un- & mapped recipes
    unmapped_recipes = ingredients.loc[ingredients['recipe_id'] == -1]['recipe'].unique()
    mapped_recipes = ingredients.loc[ingredients['recipe_id'] != -1]['recipe'].unique()

    # remove the unmapped recipes
    reduced_ingredients = ingredients[~ingredients['recipe'].isin(unmapped_recipes)]

    # get all distinct foodKG_mapped_ingredient and assign ids (ingredient_id -> foodKG_mapped_ingredient)
    ingredients_dict = {index[0]: url for index, url in
                        np.ndenumerate(reduced_ingredients['foodKG_mapped_ingredient'].unique())}
    inv_ingredients_dict = {url: index[0] for index, url in
                            np.ndenumerate(reduced_ingredients['foodKG_mapped_ingredient'].unique())}

    # use inv_ingredients_dict (foodKG_mapped_ingredient --> ingredient_id)
    reduced_ingredients['ingredient_id'] = [_lookup_index(row, inv_ingredients_dict) for row in
                                            reduced_ingredients[['foodKG_mapped_ingredient']].to_numpy().tolist()]

    # remove unnecessary columns
    reduced_ingredients.drop(['foodKG_mapped_ingredient'], axis=1, inplace=True)
    reduced_ingredients.drop(['recipe'], axis=1, inplace=True)

    return reduced_ingredients, ingredients_dict, list(mapped_recipes)


def _add_labels(row, label_dict):
    ingredient_id = row[1]
    label = row[2]

    if ingredient_id not in label_dict:
        label_dict[ingredient_id] = [label]
    else:
        label_dict[ingredient_id].append(label)


def load_ingredient_tags(graph_location, s3=None):
    """
    Loads the ingredient tag dict.

    :param graph_location: Name of the directory where the graph related data is located in.
    :param s3: The s3 connection. When the connection is not None, tries to use this connection for data loading instead
        local data loading.

    :return: The ingredient tags (ingredient_url --> tag_list)
    """
    # create empty dict
    label_dict = {}

    # read ingredient label data
    if s3 is None:
        labeled_ingredients = pd.read_csv(graph_location + 'graph_ingredient_labels.csv', sep=',').loc[:,
                              ['ingredient_name', 'ingredient_id', 'label']]
    else:
        with s3.open(graph_location + 'graph_ingredient_labels.csv') as file:
            labeled_ingredients = pd.read_csv(file, sep=',').loc[:, ['ingredient_name', 'ingredient_id', 'label']]

    # convert to dict (ingredient)
    for index, row in labeled_ingredients.iterrows():
        _add_labels(row, label_dict)

    return label_dict


def normalize_ingredients(recipes):
    """
    Normalizes all ingredients of the recipe dataframe by changing all mg units into g, and normalizing all values based
     on a recipe weight of 100g.
    :param recipes: The recipe dataframe
    :return: The normalized ingredients with a new comment row describing the type of normalization (refer to
        '_check_and_normalize_weights()')
    """
    # Init the parallel lib
    pandarallel.initialize(verbose=0)

    # Copy all ingredient values in a new data frame and change all mg-based nutrients into g-based ones. Keep cal
    # values as they are.
    normalized_ingredients = pd.DataFrame()
    normalized_ingredients[
        ['servingsPerRecipe', 'servingSize [g]', 'calories [cal]', 'caloriesFromFat [cal]', 'totalFat [g]',
         'saturatedFat [g]']] = recipes[
        ['servingsPerRecipe', 'servingSize [g]', 'calories [cal]', 'caloriesFromFat [cal]', 'totalFat [g]',
         'saturatedFat [g]']]
    normalized_ingredients['cholesterol [g]'] = recipes['cholesterol [mg]'].map(lambda x: x / 100)
    normalized_ingredients['sodium [g]'] = recipes['sodium [mg]'].map(lambda x: x / 100)
    normalized_ingredients[['totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]', 'protein [g]']] = recipes[
        ['totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]', 'protein [g]']]

    # Computation expensive (~15min on my laptop), might change to vectorized implementation...
    # Normalize all ingredients to 100g.
    normalized_ingredients = normalized_ingredients. \
        parallel_apply(lambda row: _check_and_normalize_weights(row), axis=1)
    return normalized_ingredients


def _check_and_normalize_weights(nutrient_row):
    """
    Normalize the weights via the following heuristic.

    A) Make sure that all ingredients weight less than the serving size (because this is, according to food.com, the
    scale) --> normalize to 100g
    B) The serving size is faulty. Test if the ingredients summed weight is less than 'serving size' *
    'servings per recipe'. --> normalize based on that value & mark the value with one star '*'.
    C) The serving size does not correspond to both alternatives. --> skip normalization & mark the value with two
    stars '**'.

    :param nutrient_row: A single row of the nutrient dataframe.
    :return: The normalized rows including the star-markings.
    """
    # Note: Saturated fat is part of total fat, so we do not need to take that into consideration. Also, dietary fiber
    # and sugars are both parts of total carbs, so we can also discard those.
    sum_of_ingredient_weights = sum(
        nutrient_row[['totalFat [g]', 'cholesterol [g]', 'sodium [g]', 'totalCarbohydrate [g]', 'protein [g]']])

    serving_size = nutrient_row[['servingSize [g]']][0]
    servings_per_recipe = nutrient_row[['servingsPerRecipe']][0]

    # Get factor, and if necessary, the marking.
    if serving_size >= sum_of_ingredient_weights:  # A
        normalization_function = lambda x: x * 100 / serving_size
        marking = ''
    elif serving_size * servings_per_recipe >= sum_of_ingredient_weights:  # B
        normalization_function = lambda x: (x / servings_per_recipe) * 100 / (serving_size / servings_per_recipe)
        marking = '*'
    else:  # C
        normalization_function = lambda x: x
        marking = '**'

    # Normalize values based on the factor
    nutrient_row[['totalFat [g]', 'saturatedFat [g]', 'cholesterol [g]', 'sodium [g]', 'totalCarbohydrate [g]',
                  'dietaryFiber [g]', 'sugars [g]', 'protein [g]']] \
        = nutrient_row[['totalFat [g]', 'saturatedFat [g]', 'cholesterol [g]', 'sodium [g]', 'totalCarbohydrate [g]',
                        'dietaryFiber [g]', 'sugars [g]', 'protein [g]']].map(normalization_function)

    # Add marking
    nutrient_row['normalization_comment'] = marking

    return nutrient_row
