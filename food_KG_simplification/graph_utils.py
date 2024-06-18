import math
from collections import defaultdict

import numpy as np
import pandas as pd
import rdflib
from tqdm.notebook import tqdm
import string

# Printable regex (same as printable in package string, but without whitespaces with exceptions (whitespace, newline))
valid_chars = string.digits + string.ascii_letters + r"""!#$%&()*+,-./:;=?@[]^_`{|}~""" + ' '

# ENTITIES TYPES
MAPPED_INGREDIENT = 'FoodKG_ingredient'
RECIPE = 'FoodKG_recipe'
FOOD_ON_CLASS = 'FoodOn_class'
RECIPE_INGREDIENT = 'FoodKG_recipe_ingredient'
UBERON_CLASS = 'Uberon_class'
GENEPIO_CLASS = 'Genepio_class'
TAXON_CLASS = 'NCBITTaxon_class'
OBI_CLASS = 'Obi_class'
BFO_CLASS = 'Bfo_class'
CHEBI_CLASS = 'Chebi_class'
NCIT_CLASS = 'Ncit_class'
TYPE = 'type'
LABEL = 'label'
OTHER = 'other'

# URLS
RDF_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
RDFS_LABEL = 'http://www.w3.org/2000/01/rdf-schema#label'
RDFS_PROPERTY = 'http://www.w3.org/2000/01/rdf-schema#Property'

HUMMUS_USER = 'https://www.hummus.uni-passau.de/user'
HUMMUS_REVIEW = 'https://www.hummus.uni-passau.de/review'
HUMMUS_TAG = 'https://www.hummus.uni-passau.de/tag'
HUMMUS_INGREDIENT = 'https://www.hummus.uni-passau.de/ingredient'

HUMMUS_JOINED_AT = 'https://www.hummus.uni-passau.de/joined_at'
HUMMUS_FOLLOWS_COUNT = 'https://www.hummus.uni-passau.de/follows_count'
HUMMUS_FOLLOWER_COUNT = 'https://www.hummus.uni-passau.de/follower_count'

HUMMUS_HAS_AUTHOR = 'https://www.hummus.uni-passau.de/has_author'
HUMMUS_HAS_DESCRIPTION = 'https://www.hummus.uni-passau.de/has_description'
HUMMUS_HAS_DURATION = 'https://www.hummus.uni-passau.de/has_duration'
HUMMUS_HAS_DIRECTIONS = 'https://www.hummus.uni-passau.de/has_directions'
HUMMUS_SERVES = 'https://www.hummus.uni-passau.de/serves'
HUMMUS_LAST_CHANGED_AT = 'https://www.hummus.uni-passau.de/last_changed_at'
HUMMUS_HAS_TAG = 'https://www.hummus.uni-passau.de/has_tag'
HUMMUS_AMOUNT_OF_SERVINGS = 'https://www.hummus.uni-passau.de/amount_of_servings'
HUMMUS_HAS_SERVING_SIZE = 'https://www.hummus.uni-passau.de/has_serving_size'
HUMMUS_HAS_CALORIES = 'https://www.hummus.uni-passau.de/has_calories'
HUMMUS_HAS_CALORIES_FROM_FAT = 'https://www.hummus.uni-passau.de/has_calories_from_fat'
HUMMUS_HAS_TOTAL_FAT = 'https://www.hummus.uni-passau.de/has_total_fat'
HUMMUS_HAS_SATURATED_FAT = 'https://www.hummus.uni-passau.de/has_saturated_fat'
HUMMUS_HAS_CHOLESTEROL = 'https://www.hummus.uni-passau.de/has_cholesterol_mg'
HUMMUS_HAS_SODIUM = 'https://www.hummus.uni-passau.de/has_sodium_mg'
HUMMUS_HAS_TOTAL_CARBS = 'https://www.hummus.uni-passau.de/has_total_carbohydrate'
HUMMUS_HAS_DIETARY_FIBER = 'https://www.hummus.uni-passau.de/has_dietary_fiber'
HUMMUS_HAS_SUGARS = 'https://www.hummus.uni-passau.de/has_sugars'
HUMMUS_HAS_PROTEIN = 'https://www.hummus.uni-passau.de/has_protein'
HUMMUS_HAS_REVIEW = 'https://www.hummus.uni-passau.de/has_review'
HUMMUS_AVG_RATING = 'https://www.hummus.uni-passau.de/avg_rating'
HUMMUS_NUMBER_OF_RATINGS = 'https://www.hummus.uni-passau.de/amount_of_ratings'
HUMMUS_URL = 'https://www.hummus.uni-passau.de/url'
HUMMUS_FSA = 'https://www.hummus.uni-passau.de/has_fsa_score'
HUMMUS_WHO = 'https://www.hummus.uni-passau.de/has_who_score'
HUMMUS_NUTRI = 'https://www.hummus.uni-passau.de/has_nutri_score'

HUMMUS_HAS_REVIEWER = 'https://www.hummus.uni-passau.de/has_reviewer'
HUMMUS_HAS_RATING = 'https://www.hummus.uni-passau.de/has_rating'
HUMMUS_HAS_TEXT_REVIEW = 'https://www.hummus.uni-passau.de/has_text_review'
HUMMUS_LAST_MODIFIED_AT = 'https://www.hummus.uni-passau.de/last_modified_at'
HUMMUS_HAS_LIKES = 'https://www.hummus.uni-passau.de/has_likes'


def _get_entity_type(entity_url):
    if 'http://idea.rpi.edu/heals/kb/recipe' in entity_url:
        if '/ingredient' in entity_url:
            return RECIPE_INGREDIENT
        else:
            return RECIPE
    elif 'http://idea.rpi.edu/heals/kb/ingredientname/' in entity_url:
        return MAPPED_INGREDIENT
    elif 'http://purl.obolibrary.org/obo/FOODON_' in entity_url:
        return FOOD_ON_CLASS
    elif 'http://purl.obolibrary.org/obo/UBERON_' in entity_url:
        return UBERON_CLASS
    elif 'http://purl.obolibrary.org/obo/GENEPIO_' in entity_url:
        return GENEPIO_CLASS
    elif 'http://purl.obolibrary.org/obo/NCBITaxon_' in entity_url:
        return TAXON_CLASS
    elif 'http://purl.obolibrary.org/obo/OBI_' in entity_url:
        return OBI_CLASS
    elif 'http://purl.obolibrary.org/obo/BFO_' in entity_url:
        return BFO_CLASS
    elif 'http://purl.obolibrary.org/obo/CHEBI_' in entity_url:
        return CHEBI_CLASS
    elif 'http://purl.obolibrary.org/obo/NCIT_' in entity_url:
        return NCIT_CLASS
    else:
        return OTHER


def _is_entity_valid(entity):
    if isinstance(entity, str):
        if entity.startswith('http'):  # for urls
            type = _get_entity_type(entity)
            is_valid = type == RECIPE_INGREDIENT or type == RECIPE or type == MAPPED_INGREDIENT or type == FOOD_ON_CLASS
        else:  # literals are always okay, e.g. comments, labels, ...
            is_valid = True
    else:  # Entity is blank node or plain value
        if math.isnan(entity):  # blank node
            is_valid = False
        else:
            is_valid = True
    return is_valid


def _has_valid_entities(triple):
    subject, predicate, object = triple[0], triple[1], triple[2]
    is_subject_valid = _is_entity_valid(subject)
    is_object_valid = _is_entity_valid(object)

    return is_subject_valid and is_object_valid


def reduce_graph(kg_data):
    # Calculates the valid and invalid indices
    valid_index_series = [_has_valid_entities(triple) for triple in kg_data[
        ['subject', 'predicate', 'object']].to_numpy()]

    # Keeps valid indices only
    reduced_graph = kg_data[valid_index_series]
    print(f"Removed {kg_data.shape[0] - reduced_graph.shape[0]} triplets from kg")

    return reduced_graph


def _is_invalid_triple(triple):
    subject, predicate, object = triple[0], triple[1], triple[2]
    is_invalid = 'http://idea.rpi.edu/heals/kb/uses' in predicate \
                 or 'http://idea.rpi.edu/heals/kb/ing_name' in predicate
    return is_invalid


def merge_ingredient_nodes(kg_data, graph_ingredient_mapping):
    # 1. Remove FoodKG_ingredient nodes
    valid_index_series = [not _is_invalid_triple(triple) for triple in kg_data[
        ['subject', 'predicate', 'object']].to_numpy()]
    reduced_graph = kg_data[valid_index_series]  # Keeps valid indices only
    print(f"Number of triples after removing all 'uses' and 'ing_name' relations: {reduced_graph.shape[0]}")

    # 2. Update the recipe-ingredient relation
    new_relations = pd.DataFrame()
    new_relations['subject'] = graph_ingredient_mapping[['recipe']]
    new_relations['predicate'] = 'http://idea.rpi.edu/heals/kb/uses'
    new_relations['object'] = graph_ingredient_mapping[['foodKG_mapped_ingredient']]
    print(f"Size of new relations: {new_relations.shape[0]}")
    resulting_graph = np.concatenate((reduced_graph, new_relations.to_numpy()), axis=0)
    print(f"Number of triples after adding all new 'uses' relations: {resulting_graph.shape[0]}")

    # 3. Set ingredient types to each ingredient
    ingredient_types = pd.DataFrame()
    ingredient_types['subject'] = graph_ingredient_mapping['foodKG_mapped_ingredient'].unique()
    ingredient_types['predicate'] = RDF_TYPE
    ingredient_types['object'] = HUMMUS_INGREDIENT
    print(f"Size of new relations: {ingredient_types.shape[0]}")
    resulting_graph = np.concatenate((resulting_graph, ingredient_types.to_numpy()), axis=0)
    print(f"Number of triples after adding all ingredient type relations: {resulting_graph.shape[0]}")

    result = pd.DataFrame(resulting_graph, columns=['subject', 'predicate', 'object'])
    return result


def read_triples_rdflib(kg_data):
    # This methods succeeds, but when serializing, it won't terminate. I assume I constructed the graph somehow wrong...
    print('Start converting triples...')
    graph = rdflib.Graph()

    # add default namespaces
    food_on = rdflib.Namespace('http://purl.obolibrary.org/obo/FOODON_')
    graph.bind('food_on', food_on)

    # add data
    for index, row in tqdm(kg_data.iterrows(), total=kg_data.shape[0]):
        subject = rdflib.URIRef(row[0])
        predicate = rdflib.URIRef(row[1])

        unparse_object = row[2]
        if unparse_object.startswith('http://') or unparse_object.startswith('https://'):
            object = rdflib.URIRef(unparse_object)
        else:
            object = rdflib.Literal(unparse_object)

        triple = (subject, predicate, object)
        graph.add(triple)

    return graph


def convert_to_triples(kg_data, user_data, recipe_data, review_data):
    namespaces = defaultdict(int)
    prefixes = {}

    kg_triples = _convert_data(kg_data, 'FoodKG data', namespaces, prefixes)
    user_triples = _convert_data(user_data, 'User data', namespaces, prefixes)
    recipe_triples = _convert_data(recipe_data, 'Recipe data', namespaces, prefixes)
    review_triples = _convert_data(review_data, 'Review data', namespaces, prefixes)

    print("Number of prefixes/namespaces: " + str(len(prefixes) + 1))  # +1 for xsd

    turtle_namespaces = ''
    for namespace, count in sorted(namespaces.items(), key=lambda x: x[1], reverse=True):
        prefix = prefixes[namespace]
        turtle_namespaces += f"@prefix {prefix}: <{namespace}> .\n"

    # Manually add xsd
    turtle_namespaces += "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"

    return turtle_namespaces, kg_triples, user_triples, recipe_triples, review_triples


def _convert_data(data, data_name, namespaces, prefixes):
    print("Start Converting " + data_name + "...")
    data_np = data.to_numpy()
    triples = [(_use_namespace(namespaces, prefixes, subject), _use_namespace(namespaces, prefixes, predicate),
                _use_namespace(namespaces, prefixes, obj)) for subject, predicate, obj in tqdm(data_np)]
    print("Number of triples in " + data_name + ": " + str(len(triples)))
    return triples


def write_triples(turtle_namespaces, triples, file_name, graph_location):
    skipped_triples = []

    print("Start Writing " + file_name + "...")
    with open(graph_location + file_name + '.ttl', 'w') as file:
        file.write(turtle_namespaces)
        for subject, predicate, obj in tqdm(triples, total=len(triples)):
            if subject.startswith("'"):  # invalid subjects (e.g. blank nodes) --> skip.
                skipped_triples.append(f"{subject} {predicate} {obj}")
                continue
            if isinstance(obj, list):
                obj = ', '.join(obj)[:-2]

            try:
                file.write(f"{subject} {predicate} {obj} .\n")
            except UnicodeEncodeError:
                file.write(f"{subject.encode('utf8')} {predicate.encode('utf8')} {obj.encode('utf8')} .\n")

    return skipped_triples


def _use_namespace(namespaces, prefixes, term):
    if isinstance(term, str) and term.startswith("http"):  # for urls
        split_number = 1
        spliterator = '/'
        if '#' in term:
            spliterator = '#'
        elif '/recipe/' in term:
            temp_splits = term.split("/recipe/")[1]
            split_number += temp_splits.count('/')  # count additional '/' in the recipe name
        elif 'ingredientname' in term:
            temp_splits = term.split("/ingredientname/")[1]
            split_number += temp_splits.count('/')  # count additional '/' in the ingredient name

        namespace_split = term.rsplit(spliterator, split_number)
        namespace = namespace_split[0] + spliterator
        if namespace not in prefixes:
            prefix_splits = (namespace_split[0].replace("http://", "").replace(".", "_").
                             replace("/", "_").replace("www", "").rsplit('_'))
            prefix_splits.reverse()
            all_prefixes = prefixes.values()
            for index, _ in enumerate(prefix_splits):
                potential_prefix = ''
                for i in range(index, -1, -1):  # build prefix
                    potential_prefix += prefix_splits[i] + '_'
                    # Cleans prefix, only allows asscii letters and _
                    potential_prefix = ''.join(filter(lambda x: x in string.ascii_letters + '_', potential_prefix))
                if potential_prefix not in all_prefixes:  # test if prefix is already present
                    prefixes[namespace] = potential_prefix.rsplit("_", 1)[0]
                    break
        prefix = prefixes[namespace]
        namespaces[namespace] += 1
        # clean term
        escaped_term = ''.join(map(lambda x: x + '\\/', namespace_split[1:])).rsplit('\\/', 1)[0]
        escaped_term = (escaped_term.replace('^', '\\^').replace('.', '\\.')
                        .replace('-', '\\-'))
        # make sure there are no double escaped chars
        escaped_term = escaped_term.replace('\\\\^', '\\^').replace('\\\\.', '\\.')
        term = prefix + ':' + escaped_term
    else:  # For literals
        # Split term and type if present
        xsd_type = None
        if isinstance(term, str) and '^^xsd:' in term:
            term, xsd_type = term.split('^^xsd:')
            xsd_type = '^^xsd:' + xsd_type

        # Clean strings or add type if present
        if isinstance(term, str):  # Clean strings
            printable_term = ''.join(filter(lambda x: x in valid_chars, term))
            # Note: The data cleaning step also removes chars like ", thus it must be re-appended when occurring
            # because of a xsd type.

            if xsd_type is not None:  # Add type if present
                term = '\"' + printable_term + '\"' + xsd_type
            else:
                term = '\'' + printable_term + '\''
        elif isinstance(term, float) and math.isnan(term):
            term = '\'' + str(term) + '\''
        elif xsd_type is not None:  # Add type if present for all other cases with type
            term = '\"' + str(term) + '\"' + xsd_type
    return term


def add_data(recipes_df, reviews_df, users_df, food_locator_dict, kg_data):
    print('Extract user data...')
    user_data = _extract_user_data(users_df)
    print('User data: ' + str(len(user_data)) + ' new triples')

    print('Extract recipe data...')
    recipe_data = _extract_recipe_data(recipes_df)
    print('Recipe data: ' + str(len(recipe_data)) + ' new triples')

    print('Extract review data...')
    review_data = _extract_review_data(food_locator_dict, reviews_df)
    print('Review data: ' + str(len(review_data)) + ' new triples')

    print('Convert data to pandas...')
    user_data = pd.DataFrame(user_data, columns=kg_data.columns)
    recipe_data = pd.DataFrame(recipe_data, columns=kg_data.columns)
    review_data = pd.DataFrame(review_data, columns=kg_data.columns)

    return kg_data, user_data, recipe_data, review_data


def _extract_review_data(food_locator_dict, reviews_df):
    # add all review relations once as predicate
    review_data = [[HUMMUS_HAS_REVIEWER, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_HAS_REVIEW, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_RATING, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_HAS_TEXT_REVIEW, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_LAST_MODIFIED_AT, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_HAS_LIKES, RDF_TYPE, RDFS_PROPERTY]]

    for index, row in tqdm(reviews_df.iterrows(), total=reviews_df.shape[0]):
        _append_reviews(review_data, food_locator_dict, row)

    return review_data


def _append_reviews(data, food_locator_dict, row):
    if row['rating'] < 6:  # rating of 6 is authorship, ignore those
        review_url = HUMMUS_REVIEW + '/' + str(row['review_id'])
        # review type review
        data.append([review_url, RDF_TYPE, HUMMUS_REVIEW])
        # review has_reviewer member
        data.append([review_url, HUMMUS_HAS_REVIEWER, HUMMUS_USER + '/' + str(row['member_id'])])
        # review has_rating rating
        data.append([review_url, HUMMUS_HAS_RATING, _to_integer(row['rating'])])
        # review has_review_text text
        data.append([review_url, HUMMUS_HAS_TEXT_REVIEW, _remove_link(row['text'])])
        # review has_likes likes
        data.append([review_url, HUMMUS_HAS_LIKES, _to_integer(row['likes'])])
        # review last_modified_at date
        data.append([review_url, HUMMUS_LAST_MODIFIED_AT, _to_date_time(row['last_modified_date'])])
        # recipe has_review review
        data.append([food_locator_dict[row['recipe_id']], HUMMUS_HAS_REVIEW, review_url])


def _extract_recipe_data(recipes_df):
    # add all recipe relations once as predicate
    recipe_data = [[HUMMUS_HAS_DESCRIPTION, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_DURATION, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_HAS_DIRECTIONS, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_SERVES, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_LAST_CHANGED_AT, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_TAG, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_AMOUNT_OF_SERVINGS, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_SERVING_SIZE, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_CALORIES, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_CALORIES_FROM_FAT, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_TOTAL_FAT, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_SATURATED_FAT, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_CHOLESTEROL, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_HAS_SODIUM, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_TOTAL_CARBS, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_DIETARY_FIBER, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_SUGARS, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_HAS_PROTEIN, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_HAS_AUTHOR, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_AVG_RATING, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_NUMBER_OF_RATINGS, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_URL, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_FSA, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_WHO, RDF_TYPE, RDFS_PROPERTY],
                   [HUMMUS_NUTRI, RDF_TYPE, RDFS_PROPERTY]]

    # set of all cleaned tags
    clean_tags = set()

    for index, row in tqdm(recipes_df.iterrows(), total=recipes_df.shape[0]):
        _append_recipes(recipe_data, row)

        # parse tags
        if isinstance(row['tags'], list):
            for tag in row['tags']:
                clean_tag = _remove_link(tag).replace('-', '_').replace('.', '_')
                clean_tag = HUMMUS_TAG + '/' + ''.join(
                    filter(lambda x: x in string.digits + string.ascii_letters + '_', clean_tag))
                clean_tags.add(tag)
                # add tag relations
                recipe_data.append([row['food_kg_locator'], HUMMUS_HAS_TAG, clean_tag])  # recipe has_tag tag

    # add class for all tags once
    for clean_tag in clean_tags:
        recipe_data.append([clean_tag, RDF_TYPE, HUMMUS_TAG])

    return recipe_data


def _append_recipes(data, row):
    # recipe has_description descr.
    data.append([row['food_kg_locator'], HUMMUS_HAS_DESCRIPTION, _remove_link(row['description'])])
    # recipe has_duration duration
    data.append([row['food_kg_locator'], HUMMUS_HAS_DURATION, row['duration']])
    # recipe has_directions dirs.
    data.append([row['food_kg_locator'], HUMMUS_HAS_DIRECTIONS, _remove_links(row['directions'])])
    # recipe serves #count
    data.append([row['food_kg_locator'], HUMMUS_SERVES, row['serves']])
    # recipe last_changed_at date
    data.append([row['food_kg_locator'], HUMMUS_LAST_CHANGED_AT, _to_date_time(row['last_changed_date'])])
    # recipe servings #count
    data.append([row['food_kg_locator'], HUMMUS_AMOUNT_OF_SERVINGS, _to_integer(row['servingsPerRecipe'])])
    # recipe has_servings_size #g
    data.append([row['food_kg_locator'], HUMMUS_HAS_SERVING_SIZE, _to_decimal(row['servingSize [g]'])])
    # recipe has_calories #cal
    data.append([row['food_kg_locator'], HUMMUS_HAS_CALORIES, _to_decimal(row['calories [cal]'])])
    # recipe has_calories_from_fat #cal
    data.append([row['food_kg_locator'], HUMMUS_HAS_CALORIES_FROM_FAT, _to_decimal(row['caloriesFromFat [cal]'])])
    # recipe has_total_fat #g
    data.append([row['food_kg_locator'], HUMMUS_HAS_TOTAL_FAT, _to_decimal(row['totalFat [g]'])])
    # recipe has_saturated_fat #g
    data.append([row['food_kg_locator'], HUMMUS_HAS_SATURATED_FAT, _to_decimal(row['saturatedFat [g]'])])
    # recipe has_cholesterol #mg
    data.append([row['food_kg_locator'], HUMMUS_HAS_CHOLESTEROL, _to_decimal(row['cholesterol [mg]'])])
    # recipe has_sodium #mg
    data.append([row['food_kg_locator'], HUMMUS_HAS_SODIUM, _to_decimal(row['sodium [mg]'])])
    # recipe has_total_carbs #g
    data.append([row['food_kg_locator'], HUMMUS_HAS_TOTAL_CARBS, _to_decimal(row['totalCarbohydrate [g]'])])
    # recipe has_dietary_fiber #g
    data.append([row['food_kg_locator'], HUMMUS_HAS_DIETARY_FIBER, _to_decimal(row['dietaryFiber [g]'])])
    # recipe has_sugars #g
    data.append([row['food_kg_locator'], HUMMUS_HAS_SUGARS, _to_decimal(row['sugars [g]'])])
    # recipe has_protein #g
    data.append([row['food_kg_locator'], HUMMUS_HAS_PROTEIN, _to_decimal(row['protein [g]'])])
    # recipe has_author author
    data.append([row['food_kg_locator'], HUMMUS_HAS_AUTHOR, HUMMUS_USER + '/' + str(row['author_id'])])
    # recipe has_avg_rating
    data.append([row['food_kg_locator'], HUMMUS_AVG_RATING, _to_decimal(row['average_rating'])])
    # recipe has_amount_of_ratings
    data.append([row['food_kg_locator'], HUMMUS_NUMBER_OF_RATINGS, _to_integer(row['number_of_ratings'])])
    # recipe food.com url
    data.append([row['food_kg_locator'], HUMMUS_URL, row['recipe_url']])
    # recipe fsa score
    data.append([row['food_kg_locator'], HUMMUS_FSA, _to_decimal(row['fsa_score'])])
    # recipe who score
    data.append([row['food_kg_locator'], HUMMUS_WHO, _to_decimal(row['who_score'])])
    # recipe simplified nutri score
    data.append([row['food_kg_locator'], HUMMUS_NUTRI, row['nutri_score']])


def _extract_user_data(users_df):
    # add all user relations once as predicate
    user_data = [[HUMMUS_JOINED_AT, RDF_TYPE, RDFS_PROPERTY], [HUMMUS_FOLLOWS_COUNT, RDF_TYPE, RDFS_PROPERTY],
                 [HUMMUS_FOLLOWER_COUNT, RDF_TYPE, RDFS_PROPERTY]]

    for index, row in tqdm(users_df.iterrows(), total=users_df.shape[0]):
        _append_users(user_data, row)

    return user_data


def _append_users(data, row):
    user_url = HUMMUS_USER + '/' + str(row['member_id'])
    # member type user
    data.append([user_url, RDF_TYPE, HUMMUS_USER])
    # member label member_name
    data.append([user_url, RDFS_LABEL, _remove_link(row['member_name'])])
    # member joined_at date
    data.append([user_url, HUMMUS_JOINED_AT, _to_date_time(row['member_joined'])])
    # member follows_count #count
    data.append([user_url, HUMMUS_FOLLOWS_COUNT, _to_integer(row['follows_count'])])
    # member follower_count #count
    data.append([user_url, HUMMUS_FOLLOWER_COUNT, _to_integer(row['follow_me_count'])])


def _remove_links(string_list):
    return [_remove_link(s) for s in string_list]


def _remove_link(obj):
    if isinstance(obj, str):
        if obj.startswith('http://'):
            return obj[7:]
        elif obj.startswith('https://'):
            return obj[8:]
        else:
            return obj
    else:
        return obj


def _to_decimal(decimal):
    return '"' + str(decimal) + '"^^xsd:decimal'


def _to_integer(integer):
    return '"' + str(integer) + '"^^xsd:int'


def _to_date_time(date_time):
    return '"' + str(date_time).replace(' ', "T") + '"^^xsd:dateTime'
