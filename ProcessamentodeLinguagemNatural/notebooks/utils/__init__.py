import mlflow
import os

os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'
os.makedirs("../models/mlruns", exist_ok=True)
mlflow.set_tracking_uri("../models/mlruns")

from .embeddings_binary_classification.train import PyCaretEmbeddingClassificationTrainer
from .embeddings_binary_classification.evaluation import BinaryClassificationEvaluator
from .local_llm import LocalLLM