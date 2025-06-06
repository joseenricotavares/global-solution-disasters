from .nlp import CompositeClassifier, extract_entities
from .extras import DisasterBot, Prioritizer, extract_embedding_model_zip_if_needed
from sentence_transformers import SentenceTransformer

import os
p = os.path.abspath(__file__)
while os.path.basename(p) != 'src': p = os.path.dirname(p)
DIR = os.path.join(os.path.dirname(p), "models", "sbert", "gte-small.zip")



class OnlineDisasterMessagePipeline:
    """
    Pipeline for making a single prediction and extracting entities from a disaster-related message,
    and providing an informative response via DisasterBot based on predicted disaster type and urgency.
    """

    def __init__(self):
        """
        Initialize the classifier and embedding model used for predictions.
        """
        sbert_path = extract_embedding_model_zip_if_needed(DIR)
        sbert = SentenceTransformer(sbert_path, device="cpu")
        self.classifier = CompositeClassifier(sbert)
        self.prioritizer = Prioritizer()

    def predict(self, text: str) -> dict:
        """
        Perform a full pipeline prediction on the input text, including disaster classification,
        urgency prediction, entity extraction, and generating a DisasterBot response.

        Args:
            text (str): The input message related to a disaster.

        Returns:
            dict: A dictionary containing the original text, predictions, entities,
                  and the DisasterBot response string.
        """
        self._validate_input(text)

        prediction_row = self._get_prediction(text)
        entities = self._extract_entities(text)

        disasterbot_response = self._get_disasterbot_response(prediction_row)

        return {
            "text": text,
            "predictions": prediction_row,
            "entities": entities,
            "disasterbot_response": disasterbot_response
        }

    def _validate_input(self, text: str):
        """
        Validates that the input is a string.

        Args:
            text (str): Input text.

        Raises:
            ValueError: If input is not a string.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

    def _get_prediction(self, text: str) -> dict:
        """
        Gets prediction from the classifier as a dictionary, including urgency score from the prioritizer.

        Args:
            text (str): Input text.

        Returns:
            dict: Prediction results from the classifier.
        """
        prediction_df = self.classifier.predict(text)

        # Add urgency score
        prediction_df["Urgencia_Score"] = self.prioritizer.score(prediction_df)

        return prediction_df.iloc[0].to_dict()

    def _extract_entities(self, text: str) -> list:
        """
        Extract entities from the input text.

        Args:
            text (str): Input text.

        Returns:
            list: List of extracted entities, empty if none found.
        """
        entities_list = extract_entities(text)
        return entities_list[0] if entities_list else []

    def _get_disasterbot_response(self, prediction_row: dict) -> str:
        """
        Generate the DisasterBot response based on predicted disaster and urgency.

        Args:
            prediction_row (dict): Prediction dictionary including keys
                                   'Desastre_Predicted' and 'Urgencia_Predicted'.

        Returns:
            str: DisasterBot answer string or error message.
        """
        disaster_predicted = prediction_row.get('Desastre_Predicted')
        urgency_predicted = prediction_row.get('Urgencia_Predicted')

        if not disaster_predicted or not urgency_predicted:
            return ""

        # Clean the predicted disaster string to match DisasterBot's expected format
        event_type = self._normalize_disaster_event_type(disaster_predicted)
        urgency = urgency_predicted  # DisasterBot handles urgency normalization internally

        try:
            bot = DisasterBot(event_type=event_type, urgency=urgency)
            return bot.answer()
        except Exception as e:
            return f"Erro ao gerar resposta do DisasterBot: {e}"

    @staticmethod
    def _normalize_disaster_event_type(raw_event_type: str) -> str:
        """
        Normalize the predicted disaster event type string to match DisasterBot input requirements.
        E.g. converts 'Evento_Geodinâmico_Extremo' -> 'Geodinâmico'.

        Args:
            raw_event_type (str): Raw event type string from predictions.

        Returns:
            str: Normalized event type.
        """
        cleaned = raw_event_type.replace('Evento_', '').replace('_Extremo', '')
        return cleaned.capitalize()
