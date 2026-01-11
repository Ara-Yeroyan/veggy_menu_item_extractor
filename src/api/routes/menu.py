import logging
import uuid
from typing import Annotated, Union

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Body
from langsmith import traceable

from src.api.schemas import (
    ProcessMenuResponse,
    NeedsReviewResponse,
    ProcessMenuRequest,
    ClassifiedItemSchema,
    UncertainItemSchema,
    DetailedItemSchema,
)
from src.api.services import OCRService, MenuParser, MCPClient
from configs import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/process-menu", tags=["menu"])


def get_ocr_service() -> OCRService:
    return OCRService()


def get_parser() -> MenuParser:
    return MenuParser()


@router.post("", response_model=Union[ProcessMenuResponse, NeedsReviewResponse])
@traceable(name="process_menu_endpoint")
async def process_menu_multipart(
    files: Annotated[list[UploadFile], File(description="Menu images (1-5)")],
    ocr_service: Annotated[OCRService, Depends(get_ocr_service)],
    parser: Annotated[MenuParser, Depends(get_parser)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Process menu images and return vegetarian items with total sum.

    Accepts 1-5 menu images via multipart/form-data upload.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Processing {len(files)} menu images")
    
    if not files or len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Please provide between 1 and 5 menu images"
        )
    
    image_bytes_list = []
    for file in files:
        content = await file.read()
        image_bytes_list.append(content)
    
    return await _process_images(
        image_bytes_list, request_id, ocr_service, parser, settings
    )


@router.post("/base64", response_model=Union[ProcessMenuResponse, NeedsReviewResponse])
@traceable(name="process_menu_base64_endpoint")
async def process_menu_base64(
    request: ProcessMenuRequest,
    ocr_service: Annotated[OCRService, Depends(get_ocr_service)],
    parser: Annotated[MenuParser, Depends(get_parser)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Process menu images and return vegetarian items with total sum.

    Accepts 1-5 menu images as base64 encoded strings.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Processing {len(request.images)} base64 images")
    
    return await _process_images(
        request.images, request_id, ocr_service, parser, settings
    )


@traceable(name="process_images_pipeline")
async def _process_images(
    images: list,
    request_id: str,
    ocr_service: OCRService,
    parser: MenuParser,
    settings: Settings,
) -> Union[ProcessMenuResponse, NeedsReviewResponse]:
    """
    Core processing pipeline for menu images.

    Parameters
    ----------
    images : list
        List of image data (bytes or base64 strings)
    request_id : str
        Unique request identifier for tracing
    ocr_service : OCRService
        OCR service instance
    parser : MenuParser
        Menu parser instance
    settings : Settings
        Application settings

    Returns
    -------
    ProcessMenuResponse or dict
        Processing results or HITL review card
    """
    all_items = []
    for idx, img in enumerate(images):
        logger.info(f"[{request_id}] OCR processing image {idx + 1}")
        text = ocr_service.extract_text(img)
        logger.debug(f"[{request_id}] Image {idx + 1} OCR result: {text[:200]}...")
        
        items = parser.parse(text)
        for item in items:
            item.source_image = idx + 1
        all_items.extend(items)
    
    logger.info(f"[{request_id}] Parsed {len(all_items)} items from {len(images)} images")
    menu_items = all_items
    
    if not menu_items:
        logger.warning(f"[{request_id}] No menu items extracted")
        return ProcessMenuResponse(
            request_id=request_id,
            vegetarian_items=[],
            total_sum=0.0,
            status="success"
        )
    
    logger.info(f"[{request_id}] Found {len(menu_items)} menu items, calling MCP")
    
    mcp_client = MCPClient(request_id=request_id)
    result = await mcp_client.classify_and_calculate(menu_items)
    
    all_items_detailed = [
        DetailedItemSchema(**item)
        for item in result.get("all_items", [])
    ]
    
    if result.get("status") == "needs_review":
        return _build_hitl_response(request_id, result, all_items_detailed)
    
    vegetarian_items = [
        ClassifiedItemSchema(**item) 
        for item in result.get("vegetarian_items", [])
    ]
    
    return ProcessMenuResponse(
        request_id=request_id,
        vegetarian_items=vegetarian_items,
        total_sum=result.get("total_sum", 0.0),
        status="success",
        all_items=all_items_detailed
    )


def _build_hitl_response(
    request_id: str, 
    result: dict,
    all_items: list[DetailedItemSchema]
) -> NeedsReviewResponse:
    """Build HITL review response for uncertain items."""
    return NeedsReviewResponse(
        request_id=request_id,
        uncertain_items=[
            UncertainItemSchema(**item)
            for item in result.get("uncertain_items", [])
        ],
        confident_items=[
            ClassifiedItemSchema(**item)
            for item in result.get("confident_items", [])
        ],
        partial_sum=result.get("partial_sum", 0.0),
        all_items=all_items
    )
