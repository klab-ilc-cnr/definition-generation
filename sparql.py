from SPARQLWrapper import SPARQLWrapper, JSON, QueryResult
from complit_generation import *
from collections import OrderedDict
from generate_defs import parse_usems
from utility import save_to_pickle
import os
from dotenv import load_dotenv

def sparql_query_execute(query_sparql: str) -> "QueryResult.ConvertResult":

    load_dotenv()

    sparql = SPARQLWrapper(
        endpoint = os.getenv("SPARQL_REPO") # type: ignore
    )

    sparql.setReturnFormat(JSON)
    sparql.setQuery(query=query_sparql)

    try:
        ret = sparql.queryAndConvert()
    except Exception as e:
        print("Exception: ",e)
    return ret


def first_level_query(input_path) -> list[LexicalEntry]:
    """Retrieves data from SPARQL and transforms them into LexicalEntry

    Parameters:
        input_path (str): path to the query SPARQL file
    
    Returns:
        lexical_entries (list[LexicalEntry]): list of LexicalEntry partially initialized
    """
    with open(input_path, 'r') as input_file:
        query = input_file.read()
    ret = sparql_query_execute(query)

    json_senses = ret["results"]["bindings"] # type: ignore
    lexical_entries: list[LexicalEntry] = []
    if len(json_senses) > 0:
        #print("json_senses: {}".format(json_senses))
        grouped = OrderedDict()
        for item in json_senses:
            lid = item['le']['value'] # type: ignore #MUS
            if lid not in grouped:
                grouped[lid] = {
                    'lemma_id': lid,
                    'lemma': item.get('lemma',{}).get('value'),
                    'senses': []
                }
            grouped[lid]['senses'].append({
                'usem': item.get('sense',{}).get("value"),
                'definition': item.get("definition",{}).get("value"),
                'relations': [],
                'template': item.get('template',{}).get('value'),
                'example': item.get('example',{}).get('value'),
                'ai_definitions': []
                })
            grouped_data = list(grouped.values())
            #
            # print(grouped[lid])
        for entry in grouped_data:
            lexical_extry = LexicalEntry(entry['lemma'], entry['lemma_id'], parse_usems(entry['senses']))
            lexical_entries.append(lexical_extry)
    else:
        print("WARN: No Lexical Entries found! for the query: \"{}\"".format(query[:100]))
    #print([le.to_dict() for le in lexical_entries])

    return lexical_entries


def second_level_query(input_path:str, sense_id:str) -> list[Relation]:
# def second_level_query(test) -> list[Relation]:
    """Retrieves and returns all relations for a sense

    Parameters:
        input_path (str): path to the query SPARQL file
        sense_id (str): sense unique identifier

    Returns:
        relations (list[Relation]): list of relations for a sense
    """
    with open(input_path,'r') as input_file:
        query = input_file.read().replace("#USEM#", sense_id)
    ret = sparql_query_execute(query)
    json_relations = ret["results"]["bindings"] # type: ignore
    # json_relations = test["results"]["bindings"]
    relations: list[Relation] = []
    for relation in json_relations:
        relations.append(Relation(
            usem=relation.get('target', {}).get('value'),
            lemma=relation.get('lemma', {}).get('value'),
            definition=relation.get('def', {}).get('value'),
            type=relation.get('relation', {}).get('value'),
            example=relation.get('example', {}).get('value')
        ))
    return relations
    