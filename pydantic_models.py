from pydantic import BaseModel, Field
import uuid


class Definition(BaseModel):
    """Pydantic class to describe an extended definition output"""
    usem: str = Field(description="Identifier of the old definition")
    old_definition: str = Field(description="Old definition to be expanded")
    id: str = str(uuid.uuid4())
    definition: str = Field(description="Definition expanded by the model")


class Expansion(BaseModel):
    """Pydantic class to describe the list of extended definitions (for a lemma)"""
    definitions: list[Definition] = Field(description="list of extended definitions")

class DefOnly(BaseModel):
    definition: str = Field(description="Definizione del senso di una parola")


class ScoreOnly(BaseModel):
    score: int = Field(description="La valutazione data alla definizione come punteggio da 1 a 10")

class Scores(BaseModel):
    scores: list[int] = Field(description="La valutazione data alla definizione come punteggio da 1 a 10")