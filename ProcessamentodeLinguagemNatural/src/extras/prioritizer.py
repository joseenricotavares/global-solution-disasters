import pandas as pd

class Prioritizer:
    """
    Calculates a urgency score based on classifier outputs.
    """

    def __init__(self):
        # Define weights for each urgency level
        self.weights = {
            "Muito_baixa": 0.0,
            "Baixa": 0.25,
            "Moderada": 0.5,
            "Alta": 0.75,
            "Crítica": 1.0
        }
        self.urgency_levels = list(self.weights.keys())

    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate urgency score from classifier output DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with columns like:
                Urgencia_Muito_baixa, ..., Urgencia_Crítica, Urgencia_Predicted

        Returns:
            pd.Series: Urgency scores from 0 to closely 1
        """
        scores = []

        for _, row in df.iterrows():
            predicted_class = row["Urgencia_Predicted"]
            predicted_weight = self.weights.get(predicted_class, 0)

            # Get the class distribution
            class_scores = [row.get(f'Urgencia_{level}', 0) for level in self.urgency_levels]
            
            # Weighted average score
            weighted_score = sum(score * self.weights[level] for score, level in zip(class_scores, self.urgency_levels))

            # Optionally blend: 70% weight to predicted class, 30% to distribution
            final_score = 0.7 * predicted_weight + 0.3 * weighted_score
            scores.append(final_score)

        return pd.Series(scores, name="Urgencia_Score")