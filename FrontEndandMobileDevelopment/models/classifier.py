import joblib
import pandas as pd
import numpy as np
import os

p = os.path.abspath(__file__)
while os.path.basename(p) != 'models': p = os.path.dirname(p)
model_path = os.path.join(p, "selected_model.pkl")
encoder_path = os.path.join(p, "one-hot-encoder.pkl")

class RiskClassifier:
    def __init__(self, model_path: str=model_path, encoder_path: str=encoder_path):
        """
        Initializes the classifier by loading the pre-trained model and encoder.
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = self._load_model()
        self.encoder = self._load_encoder()
        self.expected_columns = self.model.feature_names_in_

    def _load_model(self):
        """
        Loads the trained classification model from disk.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        return joblib.load(self.model_path)

    def _load_encoder(self):
        """
        Loads the fitted OneHotEncoder from disk.
        """
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder not found at {self.encoder_path}")
        return joblib.load(self.encoder_path)

    def _encode_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical variables using the loaded OneHotEncoder.
        """
        # Identify categorical columns based on encoder fitting
        object_cols = self.encoder.feature_names_in_
        
        # Apply encoding
        encoded = self.encoder.transform(df[object_cols])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=self.encoder.get_feature_names_out(object_cols), 
            index=df.index
        )
        
        # Drop original categorical columns and concatenate encoded
        df_numeric = df.drop(columns=object_cols)
        df_final = pd.concat([df_numeric, encoded_df], axis=1)
        return df_final

    def _validate_features(self, df: pd.DataFrame):
        """
        Ensures that the input dataframe has the expected columns in the correct order,
        allowing 'target' to be missing if it's part of the expected columns.
        """
        expected = set(self.expected_columns)
        input_cols = set(df.columns)

        # Ignore 'target' in missing columns check (this part was added for debugging the feature_names_in_  method for LinearDiscriminantAnalysis)
        missing = expected - input_cols - {"AffectedCountAbove1000"}
        extra = input_cols - expected

        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        # Remove extra columns
        df = df[list(expected & input_cols)]

        ordered_columns = [col for col in self.expected_columns if col in df.columns]
        return df[ordered_columns]
        

    def predict_online(self, input_dict: dict) -> dict:
        """
        Receives a dictionary for a single input and returns predicted label and probability.
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_dict])
        
        # Encode and validate
        df_encoded = self._encode_input(df)
        df_validated = self._validate_features(df_encoded)

        # Predict
        probas = self.model.predict_proba(df_validated)[0]
        label = self.model.predict(df_validated)[0]
        
        return {
            "predicted_label": label,
            "probability_score": probas.tolist()  # [prob_class_0, prob_class_1]
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Receives a batch of inputs as DataFrame and returns predicted labels and probabilities.
        """
        # Encode and validate
        df_encoded = self._encode_input(df)
        df_validated = self._validate_features(df_encoded)

        # Predict
        probas = self.model.predict_proba(df_validated)
        labels = self.model.predict(df_validated)

        # Return results
        results = df.copy()
        results["predicted_label"] = labels
        results["probability_class_0"] = probas[:, 0]
        results["probability_class_1"] = probas[:, 1]
        return results
