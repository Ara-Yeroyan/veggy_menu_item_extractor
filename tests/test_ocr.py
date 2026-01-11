import pytest
from unittest.mock import Mock, patch
from PIL import Image
import io
import base64


class TestOCRService:

    def test_load_image_from_pil(self):
        from src.api.services.ocr import OCRService
        
        ocr = OCRService()
        img = Image.new("RGB", (100, 100), color="white")
        
        result = ocr._load_image(img)
        
        assert isinstance(result, Image.Image)

    def test_load_image_from_bytes(self):
        from src.api.services.ocr import OCRService
        
        ocr = OCRService()
        img = Image.new("RGB", (100, 100), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        result = ocr._load_image(buffer.getvalue())
        
        assert isinstance(result, Image.Image)

    def test_load_image_from_base64(self):
        from src.api.services.ocr import OCRService
        
        ocr = OCRService()
        img = Image.new("RGB", (100, 100), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        
        result = ocr._load_image(b64)
        
        assert isinstance(result, Image.Image)

    def test_preprocess_converts_to_grayscale(self):
        from src.api.services.ocr import OCRService
        
        ocr = OCRService()
        img = Image.new("RGB", (100, 100), color="white")
        
        result = ocr._preprocess(img)
        
        assert result.mode == "L"
