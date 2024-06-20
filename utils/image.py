"""Image utilities."""
from PIL import Image


def is_valid_image(fpath: str) -> bool:
    """Validate image data.

    Args:
        fpath (str): Image file path.

    Returns:
        bool: If True, image is valid.
    """
    try:
        Image.open(fpath).convert("RGB")
        return True
    except Exception:
        return False
    
def get_img_size(fpath: str):
    """Returns image size.

    Args:
        fpath (str): Image file path.

    Returns:
        bool: If True, image is valid.
    """
    try:
        w, h = Image.open(fpath).size
        return (w, h)
    except Exception:
        return False
