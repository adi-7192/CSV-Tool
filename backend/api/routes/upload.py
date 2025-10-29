"""
CSV upload endpoints

TODO: Implement upload logic extracted from legacy app.py
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

router = APIRouter()


@router.post("/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload and process CSV file
    
    TODO: Implement full upload logic
    """
    return {
        "message": "Upload endpoint - TODO: implement",
        "filename": file.filename,
        "status": "not_implemented"
    }


@router.post("/multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    """
    Upload multiple CSV files
    
    TODO: Implement multiple file upload logic
    """
    return {
        "message": "Multiple upload endpoint - TODO: implement",
        "file_count": len(files),
        "status": "not_implemented"
    }

