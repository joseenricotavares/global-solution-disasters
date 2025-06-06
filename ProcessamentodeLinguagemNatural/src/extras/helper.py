import os
import zipfile

def extract_embedding_model_zip_if_needed(zip_path: str) -> str:
    """
    Extracts the contents of a zipped model directly into the target folder,
    ignoring any top-level folder in the zip file.

    Args:
        zip_path (str): Path to the model's .zip file.

    Returns:
        str: Path to the directory where the model was extracted.
    """
    model_name = os.path.splitext(os.path.basename(zip_path))[0]
    extract_path = os.path.join(os.path.dirname(zip_path), model_name)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                # Remove the top-level directory (e.g., 'gte-small/')
                inner_path = os.path.relpath(member.filename, model_name)
                target_path = os.path.join(extract_path, inner_path)

                if not member.is_dir():
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())

    return extract_path