import argparse
import json
from collections import OrderedDict
from complit_generation import *
from sparql import *
from utility import format_relation, config_model
import gc
import pickle
from langchain_core.output_parsers.pydantic import PydanticOutputParser
import pydantic_models
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm
import time
import sys


def parse_relations(relations_json) -> list[Relation]:
    """Parse the json array representing semantic relations to a list of related objects

    Parameters:
        relations_json: json array representing semantic relations

    Returns:
        relations (list[Relation]): list of objects representing semantic relations
    """
    results: list[Relation] = []
    if relations_json == None:
        return results
    for relation in relations_json:
        rel = Relation(relation['usem'], relation['lemma'], relation['definition'], relation['type'], relation['example'])
        results.append(rel)
    return results


def parse_scores(json_scores) -> list[Score]:
    """Parse the json array representing the evaluations by different LLMs as judges to a list of related objects

    Parameters:
        json_scores: json array representing the evaluations by different LLMs as judges

    Returns:
        scores (list[Score]): list of objects representing evaluations by different LLMs as judges
    """
    results: list[Score] = []
    if json_scores == None:
        return results
    for score in json_scores:
        ai_score = Score(score['model'], score['score'])
        results.append(ai_score)
    return results


def parse_ai_definitions(json_ai_definitions) -> list[AIDefinition]:
    """Parse the json representing ai definitions to a list of related objects

    Parameters:
        json_ai_definitions: json array of AI definitions

    Returns:
        ai_definitions (list[AIDefinition]): a list of objects describing ai definitions with evaluations
    """
    results: list[AIDefinition] = []
    if json_ai_definitions == None:
        return results
    for definition in json_ai_definitions:
        ai_def = AIDefinition(definition['model'], definition['definition'], parse_scores(definition['scores']))
        results.append(ai_def)
    return results


def parse_usems(json_usems) -> list[UsemEntry]:
    """Parse the json representing senses to a list of related objects

    Parameters:
        json_usems: json array of senses

    Returns:
        senses (list[UsemEntry]): a list of objects describing senses
    """
    results: list[UsemEntry] = []
    for sense in json_usems:
        sense = UsemEntry(sense['usem'], sense['definition'], sense['template'], sense['example'], 
                          parse_relations(sense['relations']), 
                          parse_ai_definitions(sense['ai_definitions']))
        results.append(sense)
    return results


def reading_json_complit(input_path: str) -> list[LexicalEntry]:
    """Read a json file representing CompL-it objects and return a list of UsemEntry

    Parameters:
        input_path (str): relative path to the json file
    
    Returns:
        entries (list[LexicalEntry]): list of LexicalEntry describing a list of CompL-it lexical entries with their senses
    """
    results: list[LexicalEntry] = []
    with open(input_path,'r') as file:
        data = json.load(file)
        grouped = OrderedDict()
        for item in data:
            lid = item['lemma_id']
            if lid not in grouped:
                grouped[lid] = {
                    'lemma_id': lid,
                    'lemma': item.get('lemma'),
                    'senses': []
                }
            grouped[lid]['senses'].append({
                'usem': item.get('usem'),
                'definition': item.get('definition'),
                'relations': item.get('relations'),
                'template': item.get('template'),
                'example': item.get('example'),
                'ai_definitions': item.get('ai_definitions')
                })

        grouped_data = list(grouped.values())
    for entry in grouped_data:
        lexical_extry = LexicalEntry(entry['lemma'], entry['lemma_id'], parse_usems(entry['senses']))
        results.append(lexical_extry)
    return results

def generate_definitions(lexical_entries: list[LexicalEntry], isAllSenses: bool, modelname: str, 
                         llm: BaseChatModel, exclude: str, overwriteGeneration: bool=False,
                         ):
    parser = PydanticOutputParser(pydantic_object=pydantic_models.DefOnly)
    system_role = "Sei un esperto lessicografo.\n"
    # test = []
    timestr = time.strftime("%Y%m%d-%H%M%S")
    modelname_short = modelname.split('/')[-1]

    output_filename = 'output/llm_defs-{}.txt'.format(modelname_short)
    with open(output_filename,'w') as output:
        progress_bar_le = tqdm(desc="Lexical entries",total=len(lexical_entries),leave=True)
        promptNum = 0
        for lexical_entry in lexical_entries:
            output.write("\n*** LEMMA: {} ***\n".format(lexical_entry.lemma))
            if not isAllSenses:
                definition_limits = """
Rispondi esclusivamente con JSON valido conforme allo schema fornito.
Non aggiungere testo, spiegazioni o formattazione extra.
La definizione non deve superare le 30 parole.
Non riscrivere WORD nella definizione.
Integra queste informazioni con la tua conoscenza interna per generare la definizione.\n"""
                generation_desc = "Genera la definizione del senso della WORD data utilizzando le seguenti informazioni, dove:\n"
                word_incipit = "La WORD è \"{}\":\n".format(lexical_entry.lemma)

                progress_bar_senses = tqdm(desc="Senses", total=len(lexical_entry.senses), leave=False)
                for sense in lexical_entry.senses:
                    indice = next((i for i, d in enumerate(sense.ai_definitions) if d.model == modelname), None)
                    #check if definition generated by "modelname" is already present. If yes it will be overwritten
                    if indice is None or (indice is not None and overwriteGeneration):
                        inputForPrompt = "USEM: {}"
                        if exclude != "examples":
                            inputForPrompt += "EXAMPLE: {}"
                        if exclude != "templates":
                            inputForPrompt += "CONCEPT: {}\n"
                        output.write(inputForPrompt.format(sense.usem, sense.example, sense.template))
                        information_desc = ""
                        sense_desc = ""
                        #print("EXCLUDE: {}".format(exclude))
                        if sense.example and exclude != "examples":
                            information_desc+="- EXAMPLE: è l'esempio di uso della parola con quel senso\n"
                            sense_desc+="EXAMPLE: {}\n".format(sense.example)
                        if sense.template and exclude != "templates":
                            information_desc+="- CONCEPT: è il concetto cui fa riferimento il senso della parola\n"
                            sense_desc+="CONCEPT: {}\n".format(sense.template)
                        if sense.relations and exclude != "relations":
                            information_desc+="- RELATIONS: è un lista di relazioni con altre parole di cui è data la definizione\n"
                            relations_list = []
                            for rel in sense.relations:
                                #print("REL {} : {}".format(rel.type, rel.lemma))
                                convertedRelation = format_relation(lexical_entry.lemma,rel)
                                if (convertedRelation is not None):
                                    expl = "- {}".format(format_relation(lexical_entry.lemma,rel))
                                    relations_list.append(expl)
                            if len(relations_list) > 0:
                                sense_desc+="RELATIONS: {}\n".format(";\n".join(relations_list))
                            else:
                                print("No useful relation for {}".format(sense.usem))
                        prompt_text = system_role + generation_desc + information_desc + definition_limits + word_incipit + sense_desc
                        struct_prompt = PromptTemplate(
                            template="{format_instructions}\n{query}",
                            input_variables=["query"],
                            partial_variables={"format_instructions":parser.get_format_instructions()}
                        )
                        prompt_and_model = struct_prompt|llm
                        with open("./prompts.txt","a") as prompt_file:
                            promptNum += 1
                            prompt_file.write("*** Prompt {} ***\n{}\n".format(promptNum, prompt_text))
                            prompt_file.flush()
                        #continue
                        llm_start = time.time()
                        out_resp = prompt_and_model.invoke({"query":prompt_text})
                        llm_stop = time.time()
                        try:
                            print ("OUT_RESP: {}".format(out_resp))
                            parsed_out = parser.invoke(out_resp)
                        except OutputParserException as e:
                            print("*** Error parsing output: {}".format(out_resp)) #TODO aggiungi gestione errore
                            with  open("output/errors/error-{}.json".format(modelname_short), 'a', encoding="utf-8") as error_file:
                                error_file.write("{}: ".format(str(e)))
                                error_file.write("PROMPT: {}\n".format(prompt_text))
                                error_file.write("SENSE: {}\n".format(json.dumps(sense.to_dict())))
                                error_file.flush()
                            continue

                        indice = next((i for i, d in enumerate(sense.ai_definitions) if d.model == modelname), None)

                        if indice is not None:
                            print("\nDefinition created by {} already present. Overwrite it".format(modelname))
                            sense.ai_definitions[indice] = AIDefinition(modelname, parsed_out.definition, [], 0.0)
                        else: # altrimenti la inserisco
                            sense.ai_definitions.append(AIDefinition(modelname, parsed_out.definition, [], 0.0))
                        output.write("*** RESPONSE:*** execution time: {:.2f}s\n{}\n".format((llm_stop - llm_start),llm.invoke(prompt_text).content))
                        output.flush()
                        progress_bar_senses.update()
                    else:
                        print("\nDefinition already present. Skip.")
        # print(test)
            progress_bar_le.update()
    #error_file.close()
    return lexical_entries


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action="store_true", help="If to load data from pickle")
    parser.add_argument('-m', '--modelname', type=str, help="Name of the model to be used")
    parser.add_argument('-k', "--remove", type=bool, action=argparse.BooleanOptionalAction, help="Remove all the definitions generated by model specified by -m|--modelname from the pickel file")
    parser.add_argument('-o', '--output', type=str, help="path/filename for the json computed output")
    parser.add_argument('-p', "--pickle", type=str, help="Path to the pickle file")
    parser.add_argument('-w', "--overwrite", type=bool, action=argparse.BooleanOptionalAction, help="Overwrite definitions already generated by model specified by -m|--modelname")
    parser.add_argument('-r', '--remote', type=str, help="Remote service to use LLM modelname")
    parser.add_argument('-x', '--exclude',  type=str, help="Exclude relation, examples or both in definition generation: [relations|examples|templates]")
    parser.add_argument("--lev1", type=str, help="Path to level 1 query, for senses retrieval")
    parser.add_argument("--lev2", type=str, help="Path to level 2 query, for relations retrieval")
    args = parser.parse_args()


    ####### RELAZIONI DA ESCLUDE #########
    excludedRelationsWithUsem = {"http://lexica/mylexicon#USem796entita1",
                         }
    excludedRelationsType = {"http://klab/lexicon/vocabulary/compl-it#formal",
                         "http://klab/lexicon/vocabulary/compl-it#isA",
                         "http://klab/lexicon/vocabulary/compl-it#synonym"
                         }
   
    if args.exclude and args.exclude not in ["relations","examples","templates"]:
        print("Exclude must have one of this string value 'relations' ,'examples' or 'both'")
        sys.exit(-1)

    overwriteGeneration = args.overwrite 

    if args.output:
        outputFileName = args.output
    else:
        raise(FileNotFoundError("You have to specify json output file using -o|--output flag. Example -o output/generated_defs.json"))
  
    if args.load: # -l means load the already retrieved data 
        with open(args.pickle,'rb') as data_file:
            lexical_entries = pickle.load(data_file, encoding="utf-8")
    else:
        lexical_entries = first_level_query(args.lev1)
        #print("Retrieved {} lexical entries".format(len(lexical_entries)))
        if(args.lev2):
            howManySenses = 0
            for le in lexical_entries[:]:
                for sense in le.senses[:]:
                    sense.relations = second_level_query(input_path=args.lev2, sense_id=sense.usem)
                    #print("\tRetrieved {} senses for {}".format(len(sense.relations), sense.usem))
                    for x in sense.relations[:]: #notazione per rimuovere elementi del vettore sul quale sto iterando
                        #print("\tRelation: '{}' '{}'".format(x.type, x.usem))

                        if x.usem in excludedRelationsWithUsem:
                            sense.relations.remove(x)
                            #print("\t\tremove1 '{}' '{}'".format(x.type, x.usem))
                        elif x.type in excludedRelationsType:
                            sense.relations.remove(x)
                            #print("\t\tremove2 '{}' '{}'".format(x.type, x.usem))
                        elif x.type == "http://klab/lexicon/vocabulary/compl-it#hasSemanticType": #relazione del template => la elimino
                            sense.relations.remove(x)
                            #print("\t\tremove template: '{}' '{}'".format(x.type, x.usem))
                        #else:
                        #    print("\tRelation ok: '{}' '{}'".format(x.type, x.usem))
                    if len(sense.relations) == 0:
                        print("Removing Sense {} because have not usefull relations".format(sense.usem))
                        le.senses.remove(sense)
                    #print("\tAfter pruning {} senses for {}\n".format(len(sense.relations), sense.usem))
                if (len(le.senses) == 0):
                    print("Removing {} lexical entry".format(le.lemma))
                    lexical_entries.remove(le)
                else:
                    howManySenses += len(le.senses)
            print("Total Lexical Entries: {}".format(len(lexical_entries)))
            print("Total Sense: {}".format(howManySenses))

    if args.remove:
        if args.modelname:
            model =  args.modelname
            for le in lexical_entries:
                for sense in le.senses:
                    for definition in sense.ai_definitions[:]:
                        if definition.model == model:
                            print ("Remove definition for sense: {} generated by: {}".format(sense.usem, definition.model))
                            sense.ai_definitions.remove(definition)
            print("Removing definition from {} model done.".format(model))
        sys.exit(0)
 

        save_to_pickle(args.pickle,lexical_entries)
        print("Ended retrieving lexical entries and senses")
        #sys.exit(0)

    modelname = args.modelname
    llm = config_model(remote=args.remote,modelname=modelname,temperature=0)
    les:list[LexicalEntry] = generate_definitions(lexical_entries,False,modelname,llm, args.exclude, overwriteGeneration)
    save_to_pickle(args.pickle, les)
    with open(outputFileName,'w', encoding="utf-8") as out_json:
        encoded_out = json.dumps([le_def.to_dict() for le_def in les],ensure_ascii=False, indent=3)
        out_json.write(encoded_out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        gc.collect()
