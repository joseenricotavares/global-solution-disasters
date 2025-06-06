import pandas as pd
from typing import List, Union
from sentence_transformers import SentenceTransformer
from .base_classifiers import BaseMultiClassClassifier
import os

p = os.path.abspath(__file__)
while os.path.basename(p) != 'src': p = os.path.dirname(p)
DIR = os.path.join(os.path.dirname(p), "models", "mlruns", "718505347953396541")

class DisasterClassifier(BaseMultiClassClassifier):
    """
    Classifier for disaster types (Atmospheric, Geodynamic, Hydrologic).
    """

    def __init__(self, sbert: SentenceTransformer):
        disaster_model_paths = {
            'Evento_Atmosférico_Extremo': {
                'model_path': os.path.join(DIR, "b894320e41cf4464835110f450382918", "artifacts", "model", "model.pkl"),
                'threshold': 0.60
            },
            'Evento_Geodinâmico_Extremo': {
                'model_path': os.path.join(DIR, "31a9f49cb4cd47898b21edaa75c704d9", "artifacts", "model", "model.pkl"),
                'threshold': 0.60
            },
            'Evento_Hidrológico_Extremo': {
                'model_path': os.path.join(DIR, "4aa9b667d31c4dd8b5dd33ba379c1d44", "artifacts", "model", "model.pkl"),
                'threshold': 0.60
            },
        }
        super().__init__(sbert, disaster_model_paths)

class UrgencyClassifier(BaseMultiClassClassifier):
    """
    Classifier for urgency levels (from Muito baixa to Crítica).
    """

    def __init__(self, sbert: SentenceTransformer):
        urgency_model_paths = {
        'Muito_baixa': {
            'model_path': os.path.join(DIR, "5e6b67032b7541e5a6e8d3e56adc4098", "artifacts", "model", "model.pkl"),
            'threshold': 0.60,
            'weight': 0.8
        },
        'Baixa': {
            'model_path': os.path.join(DIR, "e267fff305bc4c74a0faf650421010a5", "artifacts", "model", "model.pkl"),
            'threshold': 0.60,
            'weight': 0.8
        },
        'Moderada': {
            'model_path': os.path.join(DIR, "4e41d36f1e5745c084794a9f913a71f5", "artifacts", "model", "model.pkl"),
            'threshold': 0.60,
            'weight': 1.0

        },
        'Alta': {
            'model_path': os.path.join(DIR, "0280dce8b3f049199b2a13023f138fb9", "artifacts", "model", "model.pkl"),
            'threshold': 0.60,
            'weight': 1.0

        },
        'Crítica': {
            'model_path': os.path.join(DIR, "3908288dc7de4dff82bc2ef7e67f3454", "artifacts", "model", "model.pkl"),
            'threshold': 0.60,
            'weight': 1.0

        },
    }
        super().__init__(sbert, urgency_model_paths)


class CompositeClassifier:
    """
    Combines multiple classifiers and ensures embeddings are computed only once.
    """

    def __init__(self, sbert: SentenceTransformer):
        self.sbert = sbert
        self.disaster_clf = DisasterClassifier(sbert)
        self.urgency_clf = UrgencyClassifier(sbert)

    def predict(self, texts: Union[str, List[str]]) -> pd.DataFrame:
        """
        Predict using all internal classifiers with shared embeddings.

        Args:
            texts (Union[str, List[str]]): List of input texts.

        Returns:
            pd.DataFrame: Single DataFrame with predictions from all classifiers,
                          with column names prefixed by classifier name.
        """
        embeddings = self.disaster_clf.get_embeddings(texts)

        # Get predictions and add prefix to avoid column name collisions
        disaster_df = self.disaster_clf.predict(embeddings=embeddings).add_prefix("Desastre_")
        urgency_df = self.urgency_clf.predict(embeddings=embeddings).add_prefix("Urgencia_")

        # Reset index to ensure proper alignment before concatenation
        disaster_df.reset_index(drop=True, inplace=True)
        urgency_df.reset_index(drop=True, inplace=True)

        # Concatenate along columns
        return pd.concat([disaster_df, urgency_df], axis=1)