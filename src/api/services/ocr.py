import pytesseract
from PIL import Image
import io
import base64
import gc
import logging
from typing import Union

logger = logging.getLogger(__name__)


class OCRService:
    """
    Optical Character Recognition service using Tesseract.
    
    Extracts text from menu images with preprocessing for better accuracy.
    """

    def __init__(self, lang: str = "eng"):
        """
        Initialize OCR service.

        Parameters
        ----------
        lang : str
            Tesseract language code
        """
        self._lang = lang
        self._config = "--oem 3 --psm 6"

    def extract_text(self, image_input: Union[str, bytes, Image.Image]) -> str:
        """
        Extract text from an image.

        Parameters
        ----------
        image_input : Union[str, bytes, Image.Image]
            Base64 string, raw bytes, or PIL Image

        Returns
        -------
        str
            Extracted text from the image
        """
        image = self._load_image(image_input)
        processed = self._preprocess(image)
        
        logger.info("Running OCR extraction")
        text = pytesseract.image_to_string(
            processed,
            lang=self._lang,
            config=self._config
        )
        
        logger.debug(f"OCR extracted {len(text)} characters")
        return text.strip()

    def extract_text_batch(self, images: list) -> list[str]:
        """
        Extract text from multiple images.

        Parameters
        ----------
        images : list
            List of image inputs (base64, bytes, or PIL Image)

        Returns
        -------
        list[str]
            List of extracted text strings
        """
        results = []
        for idx, img in enumerate(images):
            logger.info(f"Processing image {idx + 1}/{len(images)}")
            text = self.extract_text(img)
            results.append(text)
        
        self._clear_memory()
        return results

    def _clear_memory(self):
        """Clear memory after batch processing."""
        gc.collect()
        logger.debug("Memory cleared after image processing")

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """
        Load image from various input formats.

        Parameters
        ----------
        image_input : Union[str, bytes, Image.Image]
            Image in base64 string, bytes, or PIL Image format

        Returns
        -------
        Image.Image
            PIL Image object
        """
        if isinstance(image_input, Image.Image):
            return image_input
        
        if isinstance(image_input, str):
            if "," in image_input:
                image_input = image_input.split(",")[1]
            image_bytes = base64.b64decode(image_input)
        else:
            image_bytes = image_input
        
        return Image.open(io.BytesIO(image_bytes))

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.

        Parameters
        ----------
        image : Image.Image
            Input PIL Image

        Returns
        -------
        Image.Image
            Preprocessed image
        """
        if image.mode != "L":
            image = image.convert("L")
        
        return image
