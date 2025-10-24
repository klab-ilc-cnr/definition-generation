from complit_generation import *
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from tqdm import tqdm
from utility import *
import argparse
import gc
import pydantic_models
import os
from dotenv import load_dotenv
import sys
import time
from random import random
import signal

def judge_sense(modelname: str, llm: BaseChatModel, lemma: str, sense:UsemEntry, error_file, exclude: str,  overwriteScores:bool=False) -> bool:
    global senseCounter
    parser = PydanticOutputParser(pydantic_object=pydantic_models.ScoreOnly)
    system_role = "Sei un esperto lessicografo."
    activity_desc = """
Rispondi esclusivamente con JSON valido conforme allo schema fornito.
Non aggiungere testo, spiegazioni o formattazione extra.
Devi valutare la bontà delle definizioni (SENSE_DEFINITION) di una parola (WORD) assegnando un voto da 1 a 10, dove 1 è pessimo e 10 perfetto, ad ogni SENSE_DEFINITION.
"""
    info_desc = "Per ogni SENSE_DEFINITION ti saranno fornite le seguenti informazioni:"
    info_desc += """
- WORD: la parola cui appartiene il senso;\n"""
    if exclude != "examples":
        info_desc += "- EXAMPLE: è l'esempio di uso della parola con quel senso;\n" if sense.example else ""
    info_desc += "- CONCEPT: è il concetto cui fa riferimento il senso della parola;\n" if sense.template else ""
    
    if exclude != "relations":
        info_desc += "- RELATIONS: è un elenco di relazioni con altre parole;\n" if sense.relations else ""
    sense_desc  = "WORD: {};\n".format(lemma)
    if exclude != "examples":
        sense_desc += "EXAMPLE: {};\n".format(sense.example) if sense.example else ""
    sense_desc += "CONCEPT: {};\n".format(sense.template) if sense.template else ""
    if exclude != "relations":
        if sense.relations:
            relations_list = []
            for rel in sense.relations:
                expl = "- {}".format(format_relation(lemma,rel))
                if expl != "- ":
                    relations_list.append(expl)
            sense_desc+="RELATIONS: {};\n".format(";\n".join(relations_list))
    #progress_ai = tqdm(desc="Ai definitions evaluation", total=len(sense.ai_definitions), leave=False)
 
    parser = PydanticOutputParser(pydantic_object=pydantic_models.Scores)

    prompt_text = system_role + activity_desc + info_desc + sense_desc
    definition_preable = "I SENSE_DEFINITION da valutare sono:\n"
    definition = ""
    for idx, ai_def in enumerate(sense.ai_definitions):
        indice = next((i for i, d in enumerate(ai_def.scores) if d.model == modelname), None) #se esiste già una valutazione con quel modello
        if indice is None or overwriteScores:
            definition += "SENSE_DEFINITION {} - \"{}\";\n".format(idx + 1,ai_def.definition)
        else:
            print("Evaluation with model {} already present. Skip".format(ai_def.scores[indice]))
            continue
    if definition != "":
        prompt_text += definition_preable + definition
    else:
        print("No sense to evaluate.\n")
        return True
    with open("judge_prompts.txt", "a") as jp:
        jp.write("**** JUDGE PROMPT {} ****\n".format(senseCounter))
        jp.write(prompt_text)
        jp.write("\n")

    struct_prompt = PromptTemplate(
                    template="{format_instructions}\n{query}",
                    input_variables=["query"],
                    partial_variables={"format_instructions":parser.get_format_instructions()}
                )
    
    prompt_and_model = struct_prompt|llm

    try:
        out_resp = prompt_and_model.invoke({"query":prompt_text})
    except Exception as e:
        str = "Error invoking LLM {}".format(e)
        error_file.write(str)
        error_file.flush()
        return False

    #print("LLM RESPONSE: {}".format(out_resp.content))
    #sys.exit(0)
    try:
        parsed_out = parser.invoke(out_resp)
        #print("Parsed OUT: {}".format(parsed_out))
    except OutputParserException:
        print("Error parsing output") 
        error_file.write(json.dumps(sense.to_dict()), ensure_ascii=False)
        error_file.flush()
        return False
    for idx, ai_def in enumerate(sense.ai_definitions):
        #print("AI_DEF before judge: {}".format(json.dumps(ai_def.to_dict()), ensure_ascii=False))
        #se esiste lo score fatto con quel modello lo sovrascrivo
        indice = next((i for i, d in enumerate(ai_def.scores) if d.model == modelname), None)
        if indice is None:
            ai_def.scores.append(Score(model=modelname,
                            score=parsed_out.scores[idx]))
        else:   
            print("Score already present: {}. Overwrite!".format(parsed_out.scores[idx]))
            ai_def.scores[indice] = Score(model=modelname,
                            score=parsed_out.scores[idx])
        #print("AI_DEF after judge: {}".format(json.dumps(ai_def.to_dict()), ensure_ascii=False))

    #progress_ai.update()
    return True


def judge_lexical_entry(modelname: str, llm: BaseChatModel, lexical_entry: LexicalEntry, error_file, exclude: str, overwriteScore: bool=False) -> bool:
    #progress_senses = tqdm(desc="Senses", total=len(lexical_entry.senses), leave=False)
    global senseCounter
    for sense in lexical_entry.senses:
        senseCounter += 1
        success = judge_sense(modelname=modelname,
                    llm=llm, 
                    lemma=lexical_entry.lemma,
                    sense=sense,
                    error_file=error_file,
                    exclude=exclude,
                    overwriteScores=overwriteScore)
        #progress_senses.update()
        if not success:
            return False
    return True

#Controlla se tutti gli score di una definizione AI generated superano la soglia (6)
def meanScore(ai_definitions:list[Score]) -> float:
    score = 0
    for i in ai_definitions:
        if i.score < 6 :
            return -1
        else:
            score+=i.score

    return (score/len(ai_definitions))

def selectBestDefinition(lexical_entries:list[LexicalEntry]) -> None:
    print("selectBestDefinition")
    excludedUsem = {"http://lexica/mylexicon#USemTH6501abbacchiatura",
                    "http://lexica/mylexicon#USemTH2014abbozzamento",
                    "http://lexica/mylexicon#USemTH6506abbozzatura",
                    "http://lexica/mylexicon#USemTH2045accestimento",
                    "http://lexica/mylexicon#USemTH4534accettore",
                    "http://lexica/mylexicon#USemTH13167colatura",
                    "http://lexica/mylexicon#USemTH2460declinamento",
                    "http://lexica/mylexicon#USemTH2612favoleggiamento",
                    "http://lexica/mylexicon#USemTH6854geminatura",
                    "http://lexica/mylexicon#USemTH2731incarceramento",
                    "http://lexica/mylexicon#USemTH3065periodizzamento",
                    "http://lexica/mylexicon#USemTH40839risciacquatura",
                    "http://lexica/mylexicon#USemTH25004sputo"} #USEM senza relazioni significative. Solo "isA entita1"
    for le in lexical_entries:
        for sense in le.senses:
            if sense.usem in excludedUsem:
                chosenDefinition = ""
                chosenModel = ""
                chosenScore = -1
            else: 
                chosenDefinition = "no definition"
                chosenScore = -1
                chosenModel = "no model"
                for ai_def in sense.ai_definitions:
                    ai_def.mean_score = meanScore(ai_def.scores) #se -1 significa scartato
                    if chosenScore < ai_def.mean_score:
                        chosenScore = ai_def.mean_score
                        chosenModel = ai_def.model
                        chosenDefinition = ai_def.definition
            if chosenScore == -1:
                print("No chosen definition for {}".format(sense.usem))
            sense.chosenAiDef = chosenDefinition
            sense.chosenAiDefModelGenerator = chosenModel
            sense.chosenAiDefScore = chosenScore
            #print(sense.ai_score)

def statistics(lexical_entries:list[LexicalEntry]) -> None:
    print("statistics")
    modelsStat = {}
    modelMeanScore = {}
    totalDefinitions = 0

    #### STATS ON CHOSEN DEFINITION
    for le in lexical_entries:
        for sense in le.senses:
            #print(sense.chosenAiDefScore)
            if sense.chosenAiDefScore != -1:
                totalDefinitions += 1
                if sense.chosenAiDefModelGenerator in modelsStat:
                    modelsStat[sense.chosenAiDefModelGenerator] += 1
                    modelMeanScore[sense.chosenAiDefModelGenerator] += sense.chosenAiDefScore
                else:
                    modelsStat[sense.chosenAiDefModelGenerator] = 1
                    modelMeanScore[sense.chosenAiDefModelGenerator] = sense.chosenAiDefScore
            else:
                print("Discarded ai_definition: {} {} {}".format(sense.chosenAiDef, sense.chosenAiDefModelGenerator, sense.chosenAiDefScore))
    for k,v in modelsStat.items():
        print("Definition by model: {} are {}/{} mean score: {:.2f}".format(k,v,totalDefinitions, modelMeanScore[k]/v ))
    
    ### STATS FOR (LLM) JUDGE
    modelsStat = {}
    totalDefinitions = 0
    for le in lexical_entries:
        for sense in le.senses:
            for aiDef in sense.ai_definitions:
                totalDefinitions += 1
                for score in aiDef.scores:
                    if score.model in modelsStat:
                        modelsStat[score.model] += score.score
                    else:
                        modelsStat[score.model] = score.score

    for k,v in modelsStat.items():
        print("Stats by model judge: {} mean score: {:.2f}".format(k,modelsStat[k]/totalDefinitions ))
    
    ### STATS FOR (LLM) GENERATOR
    modelsStat = {}
    totalScores = {}
    for le in lexical_entries:
        for sense in le.senses:
            for aiDef in sense.ai_definitions:
                genModel = aiDef.model
                sum = 0
                for score in aiDef.scores:
                    if genModel in totalScores:
                        totalScores[genModel] += 1
                    else:
                        totalScores[genModel] = 1      
                    sum += score.score
                if genModel in modelsStat:
                    modelsStat[genModel] += sum
                else:
                    modelsStat[genModel] = sum

    for k,v in modelsStat.items():
        print("Stats by model generator: {} mean score: {:.2f}".format(k,modelsStat[k]/totalScores[k] ))

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modelname', required=False, type=str, help="Name of the model used as a judge")
    parser.add_argument('-p', '--pickle', required=True, type=str, help="Path to the pickle file from which load the data")
    parser.add_argument('-o', '--output', type=str, help="path/filename for the json computed output")
    parser.add_argument('-w', "--overwrite", type=bool, action=argparse.BooleanOptionalAction, help="Overwrite scores")
    parser.add_argument('-r', '--remote', type=str, help="If the model is remote")
    parser.add_argument('-s', '--stats', type=bool, action=argparse.BooleanOptionalAction, help="Generate statistics and evaluations")
    parser.add_argument('-x', '--exclude',  type=str, help="Exclude relation, examples or both in definition generation: [relations|examples|both]")

    args = parser.parse_args()

    global senseCounter
    senseCounter = 0

    if args.exclude and args.exclude not in ["relations","examples","templates"]:
        print("Exclude must have one of this string value 'relations' ,'examples' or 'both'")
        sys.exit(-1)

    with open(args.pickle, 'rb') as pickle_input:
        lexical_entries: list[LexicalEntry] = pickle.load(pickle_input)
    overwriteScore = args.overwrite

    if args.output:
        outputFileName = args.output
    else:
        raise(FileNotFoundError("You have to specify json output file using -o|--output flag. Example -o output/generated_defs.json"))


    if args.stats:
        selectBestDefinition(lexical_entries)
        #save_to_pickle(args.pickle, lexical_entries)
        statistics(lexical_entries)

        with open(outputFileName,'w', encoding="utf-8") as out_json: #'output/lex_defs_judged.json'
                encoded_out = json.dumps([le_def.to_dict() for le_def in lexical_entries],ensure_ascii=False, indent=3)
                out_json.write(encoded_out)
    else: 
        #Scores generation   
           #judged_le: list[LexicalEntry] = []

        llm = config_model(remote=args.remote, 
                        modelname=args.modelname,
                        temperature=0)
                        
        progress_le = tqdm(desc="Lexical entries", total=len(lexical_entries), leave=True)
        error_file = open('output/errors/judge_errors_{}.json'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")), 'w', encoding='utf-8')
        try:

            for le in lexical_entries:
                if le is not None:
                    #print("{}: {}".format(le,le.to_dict()))
                    #success = judged_le.append(judge_lexical_entry(modelname=args.modelname,
                    success = judge_lexical_entry(modelname=args.modelname,
                                        llm=llm,
                                        lexical_entry=le,
                                        error_file=error_file,
                                        exclude=args.exclude,
                                        overwriteScore=overwriteScore)
                    if not success: #problema nella valutazione => salvo quello che ho fatto
                        break
                    progress_le.update()
                else:
                    print("Lexical Entry is NONE???")
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            error_file.close()
            filename, file_extension = os.path.splitext(args.pickle)
            scoresFileName = filename + "_scores" + file_extension
            #save_to_pickle(scoresFileName, lexical_entries)
            save_to_pickle(args.pickle, lexical_entries)
            with open(outputFileName,'w', encoding="utf-8") as out_json: #'output/lex_defs_judged.json'
                encoded_out = json.dumps([le_def.to_dict() for le_def in lexical_entries],ensure_ascii=False, indent=3)
                out_json.write(encoded_out)
            print("END EVALUATION")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        gc.collect()
