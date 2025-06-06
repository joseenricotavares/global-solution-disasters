import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from typing import Optional, List


class LocalLLM:
    """
    Class to load a local Large Language Model (LLM) from a directory specified via
    an environment variable, and to generate text responses from prompts.
    """

    def __init__(self, env_var_name: str, device: str = None):
        """
        Initializes the model using a path defined by an environment variable.

        Parameters:
        - env_var_name (str): Name of the environment variable that holds the local model path.
        - device (str, optional): Device to load the model on ('cpu', 'cuda', 'mps', or None for auto).
        """

        model_path = os.getenv(env_var_name)
        if not model_path:
            raise ValueError(f"The environment variable '{env_var_name}' is not set.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Determine device map and dtype
        if device is None:
            device_map = "auto"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            device_map = {"": device}
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=False
        )

        self.streamer = TextStreamer(self.tokenizer)

    def prompt(
        self,
        prompt_text: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = True,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """
        Generates a response from the model given an input prompt.

        Parameters:
        - prompt_text (str): The input prompt to send to the model.
        - max_tokens (int): Maximum number of new tokens to generate.
        - temperature (float): Controls randomness; higher values yield more creative outputs.
        - top_p (float): Nucleus sampling; considers tokens with cumulative probability up to top_p.
        - top_k (int): Limits sampling to top_k most likely tokens.
        - repetition_penalty (float): Penalizes repeated phrases in output.
        - num_beams (int): Number of beams for beam search. Set >1 for deterministic generation.
        - do_sample (bool): Whether to use sampling (True) or greedy/beam search (False).
        - stop_tokens (List[str], optional): Stop generation when any of these tokens are found.

        Returns:
        - str: The generated text.
        """
        try:
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

            generation_args = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            output_ids = self.model.generate(**inputs, **generation_args)
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Optionally truncate at stop tokens
            generated = response[len(prompt_text):].strip()
            if stop_tokens:
                for stop in stop_tokens:
                    stop_index = generated.rfind(stop) #getting not the first, but the last occurrence with rfind
                    if stop_index != -1:
                        generated = generated[:stop_index].strip()
                        break

            return generated

        except Exception as e:
            return f"[Error during generation: {str(e)}]"
    
    def generate_emergency_message( 
        self,
        tipo_catastrofe: str,
        consequencia: str,
        nivel_urgencia: str,
        descricao_urgencia: str,
        ajuda_solicitada: str,
        descricao_ajuda: str,
        prompt_path: str = None
    ) -> str:
        """
        Generates a short and impactful emergency message based on a prompt read from a file.
        
        Parameters:
        - tipo_catastrofe (str): Type of disaster (e.g., earthquake, flood).
        - consequencia (str): Consequence of the disaster.
        - nivel_urgencia (str): Urgency level of the situation.
        - descricao_urgencia (str): Description detailing the urgency.
        - ajuda_solicitada (str): Type of help requested.
        - descricao_ajuda (str): Description of the help needed.
        - prompt_path (str): File path to the prompt template text file. Defaults to "emergency_prompt.txt".
        
        Returns:
        - str: Generated emergency message or an error message if the prompt file cannot be read.
        """
        if prompt_path is None:
            prompt_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "emergency_message_prompt.txt"))

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except Exception as e:
            return f"[Error reading prompt file: {str(e)}]"

        # Format the prompt template with provided disaster and help details
        prompt_text = prompt_template.format(
            tipo_catastrofe=tipo_catastrofe,
            consequencia=consequencia,
            nivel_urgencia=nivel_urgencia,
            descricao_urgencia=descricao_urgencia,
            ajuda_solicitada=ajuda_solicitada,
            descricao_ajuda=descricao_ajuda
        )

        # Call the self.prompt method to generate the emergency message with specified parameters
        return self.prompt(
            prompt_text=prompt_text,
            max_tokens=100,              # Limit the output length to 100 tokens
            temperature=0.7,             # Control creativity/randomness in generation
            top_p=0.9,                   # Nucleus sampling parameter
            top_k=40,                    # Limits next token choices to top_k options
            repetition_penalty=1.2,      # Penalizes repetition in the output
            num_beams=1,                 # Number of beams for beam search (1 means no beam search)
            do_sample=True,              # Enable sampling to add randomness
            stop_tokens=['"']            # Stop generation when this token is encountered
        ).replace('"', '')

