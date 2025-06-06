import os

p = os.path.abspath(__file__)
while os.path.basename(p) != 'src': p = os.path.dirname(p)
DIR = os.path.join(os.path.dirname(p), "kb")

class DisasterBot:
    """
    A bot that provides information about extreme disaster events 
    based on event type and urgency level.
    """

    def __init__(self, event_type: str, urgency: str, base_dir: str = DIR):
        """
        Initializes the DisasterBot with event type, urgency, and base directory.
        
        Args:
            event_type (str): Type of event, e.g. 'Atmosférico', 'Geodinâmico', 'Hidrológico'
            urgency (str): Urgency level, e.g. 'Muito_baixa', 'Baixa', 'Moderada', 'Alta', 'Crítica'
            base_dir (str): Base directory where the text files are stored
        """
        self._raw_event_type = event_type
        self._raw_urgency = urgency
        self._base_dir = base_dir

    def answer(self) -> str:
        """
        Returns the content of the extreme event file after validating inputs.

        Returns:
            str: Content of the corresponding text file

        Raises:
            FileNotFoundError: If the file is not found
            ValueError: If any input is invalid
        """
        try:
            event_type = self._normalize_event_type()
            urgency = self._normalize_urgency()
            self._validate_event_type(event_type)
            self._validate_urgency(urgency)
            filename = self._build_filename(event_type, urgency)
            filepath = self._build_filepath(filename)
            return self._read_file(filepath)
        except (ValueError, FileNotFoundError):
            return (
                "Desculpe, não conseguimos identificar o tipo de evento e/ou o nível de urgência do seu caso. Contate-nos novamente."
            )


    def _normalize_event_type(self) -> str:
        """Capitalizes the event type for standardization."""
        return self._raw_event_type.capitalize()

    def _normalize_urgency(self) -> str:
        """Capitalizes urgency unless it's 'Muito_baixa', which stays as-is."""
        return self._raw_urgency if self._raw_urgency == 'Muito_baixa' else self._raw_urgency.capitalize()

    def _validate_event_type(self, event_type: str):
        """Checks if the event type is valid."""
        valid = ['Atmosférico', 'Geodinâmico', 'Hidrológico']
        if event_type not in valid:
            raise ValueError(f"Invalid event type: {event_type}. Valid options: {valid}")

    def _validate_urgency(self, urgency: str):
        """Checks if the urgency is valid."""
        valid = ['Muito_baixa', 'Baixa', 'Moderada', 'Alta', 'Crítica']
        if urgency not in valid:
            raise ValueError(f"Invalid urgency: {urgency}. Valid options: {valid}")

    def _build_filename(self, event_type: str, urgency: str) -> str:
        """Builds the expected filename from event type and urgency."""
        return f"Evento_{event_type}_Extremo_{urgency}.txt"

    def _build_filepath(self, filename: str) -> str:
        """Constructs the full path to the file."""
        return os.path.join(self._base_dir, filename)
    
    def _read_file(self, filepath: str) -> str:
        """Reads the content of the file from the given path."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
