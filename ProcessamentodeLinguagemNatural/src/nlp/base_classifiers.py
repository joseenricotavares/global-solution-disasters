import pandas as pd
from typing import List, Union, Dict, Tuple
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

class BaseBinaryClassifier:
    """
    Base class for binary classification tasks using sentence embeddings and sklearn models.
    """

    def __init__(self, sbert: SentenceTransformer):
        """
        Initialize with a pre-loaded SentenceTransformer model.
        
        Args:
            sbert (SentenceTransformer): A pre-trained sentence transformer model.
        """
        self.sbert = sbert

    def get_embeddings(self, texts: Union[str, List[str]]) -> pd.DataFrame:
        """
        Generate embeddings for a string or list of strings.

        Args:
            texts (Union[str, List[str]]): Input text(s) to embed.

        Returns:
            pd.DataFrame: DataFrame of embeddings (one row per input).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.sbert.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # Create a DataFrame with column names like CLS0, CLS1, ... in respect to our trained models
        return pd.DataFrame(embeddings, columns=[f"CLS{i}" for i in range(len(embeddings[0]))])

    def run_inference(self, model_path: str, embeddings: pd.DataFrame) -> pd.DataFrame:
        """
        Run prediction on given embeddings using a saved sklearn model.

        Args:
            model_path (str): Path to the `.pkl` model file.
            embeddings (pd.DataFrame): Embeddings to classify.

        Returns:
            pd.DataFrame: DataFrame with predictions and class probabilities.
        """
        model = joblib.load(model_path)
        preds = model.predict(embeddings)
        probs = model.predict_proba(embeddings)

        return pd.DataFrame({
            'Label': preds,
            'Score_0': probs[:, 0],
            'Score_1': probs[:, 1]
        })


class BaseMultiClassClassifier(BaseBinaryClassifier):
    """
    Handles multiclass classification by combining multiple binary classifiers (one-vs-rest).
    Allows per-class thresholds, weights, and fallback to 'Não_Identificado' when none is confident.
    """

    def __init__(self, sbert: SentenceTransformer, class_configs: Dict[str, Dict]):
        """
        Args:
            sbert (SentenceTransformer): Pre-trained sentence transformer model.
            class_configs (Dict): Dictionary where each key is a class label, and value is a dict with:
                - 'model_path' (str): Path to the binary model.
                - 'threshold' (float, optional): Minimum score to accept this class.
                - 'weight' (float, optional): Multiplier for the class score.
        """
        super().__init__(sbert)
        self.class_configs = class_configs

    def predict(self, texts: Union[str, List[str]] = None, embeddings: pd.DataFrame = None) -> pd.DataFrame:
        """
        Predict the class for each input and return class probabilities.
        Either texts or precomputed embeddings must be provided.

        Args:
            texts (Union[str, List[str]]): Input text(s) to classify.
            embeddings (pd.DataFrame): Optional precomputed embeddings.

        Returns:
            pd.DataFrame:
                - One column per class with Score_1 (positive class probability).
                - Predicted: class with highest score above threshold.
                - Confidence: score of the predicted class.
        """
        if embeddings is None:
            if texts is None:
                raise ValueError("Either texts or embeddings must be provided.")
            embeddings = self.get_embeddings(texts)

        class_scores = {}

        # Run binary classification for each class and apply weight
        for label, config in self.class_configs.items():
            score = self._apply_weight(label, config, embeddings)
            class_scores[label] = score

        scores_df = pd.DataFrame(class_scores)

        # Apply decision logic row-wise
        results = scores_df.apply(self._decide_class, axis=1, result_type="expand")
        scores_df["Predicted"] = results[0]
        scores_df["Confidence"] = results[1]

        return scores_df

    def _apply_weight(self, label: str, config: Dict, embeddings: pd.DataFrame) -> np.ndarray:
        """
        Runs binary inference for a class and applies the configured weight.

        Args:
            label (str): Class label.
            config (Dict): Configuration for the class.
            embeddings (pd.DataFrame): Input embeddings.

        Returns:
            np.ndarray: Weighted scores for the class.
        """
        model_path = config["model_path"]
        weight = config.get("weight", 1.0)

        pred_df = self.run_inference(model_path, embeddings)
        return pred_df["Score_1"].values * weight

    def _decide_class(self, row: pd.Series) -> Tuple[str, float]:
        """
        Determines the predicted class for a given row of scores.
        Applies per-class threshold. If no class meets the threshold,
        returns 'Não_Identificado' and the best score.

        Args:
            row (pd.Series): Scores for each class.

        Returns:
            Tuple[str, float]: Predicted class and its confidence score.
        """
        best_class = row.idxmax()
        best_score = row.max()
        threshold = self.class_configs[best_class].get("threshold", 0.0)

        if best_score >= threshold:
            return best_class, best_score
        else:
            return "Não_Identificado", best_score