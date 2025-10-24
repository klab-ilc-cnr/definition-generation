from complit import *
import complit_generation as gen
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_nebius import ChatNebius
from langchain_community.llms import DeepInfra
import os
from dotenv import load_dotenv
import sys
import pickle
from pydantic import SecretStr

def config_model(remote:str, modelname:str="", temperature=0) -> BaseChatModel:
    """Configure LLM model
    Parameters:
        remote (bool): use a remote model by Groq or a local model
        modelname (str): name of the model used
        temperature (float): temperature of the model
    Returns:
        llm (ChatGroq|ChatOllama): chat model ready for the prompt
    """
    load_dotenv()

    if remote is not None:
        match remote:
            case "ChatGroq":
                os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "") 
                chat = ChatGroq(model=modelname, temperature=temperature)
            case "OpenRouter":
                os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "") 
                print(os.environ["OPENROUTER_API_KEY"])
                chat = ChatOpenAI(model=modelname, temperature=temperature, 
                                  base_url='https://openrouter.ai/api/v1', api_key=SecretStr(os.environ["OPENROUTER_API_KEY"] ))
            case "ChatTogether":
                os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "") 
                chat = ChatTogether(model=modelname, temperature=temperature)
            case "ChatVenice":
                os.environ["VENICE_API_KEY"] = os.getenv("VENICE_API_KEY", "") 
                print(os.environ["VENICE_API_KEY"])
                chat = ChatOpenAI(model=modelname, temperature=temperature, 
                                  base_url='https://api.venice.ai/api/v1', api_key=SecretStr(os.environ["VENICE_API_KEY"]))
            case "ChatNebius":
                os.environ["NEBIUS_API_KEY"] = os.getenv("NEBIUS_API_KEY", "")
                chat = ChatNebius(model=modelname, temperature=temperature)
            case "ChatDeepInfra":
                os.environ["DEEPINFRA_API_KEY"] = os.getenv("DEEPINFRA_API_KEY", "")
                chat = DeepInfra(model=modelname, temperature=temperature)
            case _:
                print("Error when specify -r flag, value is invalid: {}".format(remote))
                sys.exit(-1)        
        return chat
    return ChatOllama(model=modelname, temperature=temperature)


def relation_to_string(relation: Relation):
    """
    Transform a semantic relation in a string

    Parameters:
        relation (Relation): object representing a semantic relation

    Returns:
        relation(str): string representing a semantic relation
    """
    result = ""
    type = relation.type
    target = relation.target.label
    if type=="hyponym":
        result = "{} of {}".format(type,target)
    elif type == "processVerb":
        result = "related process of {}".format(target)
    elif type == "purpose":
        result = "{}: {}".format(type,target)
    elif type == "deverbalAdjective":
        result = "deverbal adjective of {}".format(target)
    elif type == "approximateSynonym":
        result = "approximate synonym of {}".format(target)
    elif type == "resultingState":
        result = "resulting state of {}".format(target)
    return result


def format_relation(lemma: str, relation:gen.Relation)->str|None:
    type = relation.type
    if relation.lemma == "entità": #non si considerano le relationi con entita1 (superclasse di tutto)
        return None
    if type == "http://www.lexinfo.net/ontology/3.0/lexinfo#hyponym": #HYPONYM
        return "\"{}\" è iponimo di \"{}\" nel senso di \"{}\"".format(lemma, relation.lemma, relation.definition)
    elif type == "http://klab/lexicon/vocabulary/compl-it#isA": 
        return None
    elif type == "http://klab/lexicon/vocabulary/compl-it#formal":
        return None
    elif type == "http://www.lexinfo.net/ontology/3.0/lexinfo#approximateSynonym":
        return "\"{}\" è un quasi sinonimo di \"{}\" nel senso di \"{}\"".format(lemma, relation.lemma, relation.definition)
    elif type == "http://klab/lexicon/vocabulary/compl-it#synonym":
        return "\"{}\" è un sinonimo di \"{}\" nel senso di \"{}\"".format(lemma, relation.lemma, relation.definition)
    elif type == "http://www.lexinfo.net/ontology/3.0/lexinfo#hypernym":
        return "\"{}\" è iperonimo di \"{}\" nel senso di \"{}\"".format(lemma, relation.lemma, relation.definition)
    elif type == "http://klab/lexicon/vocabulary/compl-it#derivational":
        return "\"{}\" è derivato da \"{}\" nel senso di \"{}\"".format(lemma, relation.lemma, relation.definition)
    elif type == "http://klab/lexicon/vocabulary/compl-it#processVerb":
        return "\"{}\" deriva dal verbo \"{}\" nel senso di \"{}\"".format(lemma, relation.lemma, relation.definition)
    else:
        print("MISSING: {}".format(relation.type))
        return None


def save_to_pickle(save_path, objs):
    with open(save_path, 'wb') as out_file:
        pickle.dump(objs,out_file)