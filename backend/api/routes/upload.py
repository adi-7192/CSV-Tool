"""
Upload API Endpoints - Handle CSV file uploads
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any, List
import pandas as pd

from services.upload_service import process_csv_upload
from models.responses import UploadResponse
from services.database_reset import reset_database, check_database_schema

router = APIRouter()


@router.post("/csv", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(..., description="CSV file to upload")
):
    """
    Upload and process CSV file
    
    Returns:
        - ingestion_id: Unique ID for this upload
        - filename: Original filename
        - rows_uploaded: Number of rows processed
        - duplicates_removed: Number of duplicate rows removed
        - validation_report: Data quality report
        - column_mapping: Detected column mapping
    """
    
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    # Validate file size (100MB max)
    file_content = await file.read()
    
    if len(file_content) > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 100MB"
        )
    
    # Process upload
    result = process_csv_upload(file_content, file.filename)
    
    if not result.get('success'):
        raise HTTPException(
            status_code=500,
            detail=result.get('error', 'Upload processing failed')
        )
    
    # Shape response to include concise quality warnings
    return {
        'success': True,
        'rows_processed': result.get('rows_processed', result.get('rows_raw', 0)),
        'rows_inserted': result.get('rows_inserted', result.get('rows_uploaded', 0)),
        'validation_warnings': result.get('validation_warnings', []),
        'ingestion_id': result.get('ingestion_id'),
        'filename': result.get('filename'),
    }


@router.get("/history")
async def get_upload_history():
    """
    Get upload history from ingestion_log table
    
    Returns list of uploads with metadata:
    - ingestion_id
    - filename
    - uploaded_at
    - rows_inserted
    - date_range_start
    - date_range_end
    - validation_status
    """
    from core.database import execute_query, table_exists
    
    if not table_exists('ingestion_log'):
        return {"uploads": []}
    
    try:
        sql = """
        SELECT
            ingestion_id,
            filename,
            uploaded_at,
            rows_inserted,
            date_range_start,
            date_range_end,
            validation_status
        FROM ingestion_log
        ORDER BY uploaded_at DESC
        LIMIT 50
        """
        
        result_df = execute_query(sql)
        
        if result_df.empty:
            return {"uploads": []}
        
        # Convert to list of dicts
        uploads = result_df.to_dict('records')
        
        # Format dates as strings
        for upload in uploads:
            if 'uploaded_at' in upload and pd.notna(upload['uploaded_at']):
                upload['uploaded_at'] = str(upload['uploaded_at'])
            if 'date_range_start' in upload and pd.notna(upload['date_range_start']):
                upload['date_range_start'] = str(upload['date_range_start'])
            if 'date_range_end' in upload and pd.notna(upload['date_range_end']):
                upload['date_range_end'] = str(upload['date_range_end'])
        
        return {"uploads": uploads}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    """
    Upload multiple CSV files
    
    Returns:
        - results: List of upload results for each file
        - total_files: Total number of files processed
        - successful: Number of successful uploads
        - failed: Number of failed uploads
    """
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validate file type
            if not file.filename or not file.filename.endswith('.csv'):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Only CSV files are supported'
                })
                failed += 1
                continue
            
            # Read file content
            file_content = await file.read()
            
            # Validate file size
            if len(file_content) > 100 * 1024 * 1024:  # 100MB
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'File too large. Maximum size is 100MB'
                })
                failed += 1
                continue
            
            # Process upload
            result = process_csv_upload(file_content, file.filename)
            
            if result.get('success'):
                successful += 1
            else:
                failed += 1
            
            results.append(result)
        
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
            failed += 1
    
    return {
        "results": results,
        "total_files": len(files),
        "successful": successful,
        "failed": failed
    }


@router.post("/reset-database")
async def reset_upload_database():
    """
    Reset database by dropping sales table
    
    WARNING: This will delete ALL data!
    
    Use this before re-uploading CSVs to ensure standardized column names.
    
    Recommended workflow:
    1. Reset database (this endpoint)
    2. Re-upload all CSV files
    3. Verify schema with /api/upload/check-schema
    """
    result = reset_database('sales')
    
    if not result.get('success'):
        raise HTTPException(
            status_code=500,
            detail=result.get('error', 'Failed to reset database')
        )
    
    return {
        'success': True,
        'message': result.get('message', 'Database reset complete'),
        'warning': 'All data has been deleted. Re-upload CSV files to populate with standardized column names.'
    }


@router.get("/check-schema")
async def check_schema():
    """
    Check current database schema and report column naming issues
    
    Returns analysis showing:
    - Whether old column names exist ("Invoice Amount", etc.)
    - Whether standardized names exist ("revenue_amount", etc.)
    - Whether lineage columns exist (source_file, ingestion_id, loaded_at)
    - Recommendation for action
    """
    analysis = check_database_schema('sales')
    
    return analysis
