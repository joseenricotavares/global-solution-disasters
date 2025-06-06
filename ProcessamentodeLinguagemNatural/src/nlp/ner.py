from typing import Union, List
import spacy
import os
p = os.path.abspath(__file__)
while os.path.basename(p) != 'src': p = os.path.dirname(p)
DIR = os.path.join(os.path.dirname(p), "models", "ner", "pt_core_news_md")

ner_model = spacy.load(DIR)

def extract_entities(texts: Union[str, List[str]]) -> List[List[dict]]:
    """
    Extract named entities from a string or a list of strings using spaCy's NER.

    Parameters:
    -----------
    texts : Union[str, List[str]]
        A single string or a list of strings from which to extract named entities.

    Returns:
    --------
    List[List[dict]]
        A list of lists, where each inner list contains dictionaries representing entities
        found in the corresponding input text. Each dictionary has the following keys:
        - "text": the entity text
        - "label": the entity label
        - "start_char": start character position
        - "end_char": end character position
    """

    # Ensure the input is a list for consistent processing
    if isinstance(texts, str):
        texts = [texts]

    all_entities = []

    for text in texts:
        doc = ner_model(text) 
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

        all_entities.append(entities)

    return all_entities