import torch
import os
import tempfile
import gdown
from tqdm import tqdm


def download_with_progress(url, output_path):
    """
    Download a file from a URL with a progress bar using gdown.

    Args:
        url (str): The URL to download the file from.
        output_path (str): The local path to save the downloaded file.
    """
    print(f"Downloading model from {url}...")
    gdown.download(url, output_path, quiet=False)
    print(f"Model downloaded to: {output_path}")


def loadcompass(file, **kwargs):
    """
    Load a model from a file or URL.

    Args:
        file (str): Path to the model file or a URL.
        **kwargs: Additional arguments for torch.load.

    Returns:
        model: Loaded PyTorch model.
    """
    # Check if the file is a URL
    if file.startswith("http://") or file.startswith("https://"):
        # Create a temporary file to store the downloaded model
        temp_file_path = tempfile.mktemp()
        download_with_progress(file, temp_file_path)
        file = temp_file_path

    # Load the model
    model = torch.load(file, **kwargs)

    # Clean up wandb settings if present
    if hasattr(model, "with_wandb") and model.with_wandb:
        model.wandb._settings = ""

    # Update device if map_location is 'cpu'
    if kwargs.get("map_location") == "cpu":
        model.device = "cpu"

    # Remove temporary file if downloaded
    if file.startswith(tempfile.gettempdir()):
        os.remove(file)

    return model
