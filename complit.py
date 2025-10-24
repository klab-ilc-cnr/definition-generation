import json
import pickle


class ComplitElement:
    def __init__(self, id, label):
        self.id = id
        self.label = label


class ComplitEntry(ComplitElement):
    def __init__(self, id, label, usems):
        super().__init__(id,label)
        self.usems = usems


class ComplitUsem:
    def __init__(self, usemid, definition, example, relations):
        self.usemid = usemid
        self.definition = definition
        self.example = example
        self.relations = relations


class Relation:
    def __init__(self, type, target):
        self.type = type
        self.target = target


def parse_usems(usems):
    results = []
    for usem in usems:
        results.append(ComplitUsem(usem['usem'], usem['definition'], usem['example'], parse_relations(usem['relations'])))
    return results


def parse_relations(relations):
    results = []
    for relation in relations:
        results.append(Relation(relation['type'], ComplitElement(relation['target']['id'], relation['target']['label'])))
    return results