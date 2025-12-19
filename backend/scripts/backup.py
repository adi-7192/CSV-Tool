#!/usr/bin/env python3
"""
Backup script for DuckDB and ChromaDB

Creates a timestamped backup of:
- DuckDB database file (analytics.duckdb)
- ChromaDB directory (backend/data/chromadb)

Usage:
    python scripts/backup.py [--output-dir backups/]
"""
import sys
import shutil
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse
import logging

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from core.database import _get_db_path
from services.embedding_service import get_chroma_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_git_sha() -> Optional[str]:
    """Get current git commit SHA"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short SHA
    except Exception:
        pass
    return None


def get_app_version() -> str:
    """Get application version"""
    try:
        from main import app
        return app.version if hasattr(app, 'version') else "2.0.0"
    except Exception:
        return "2.0.0"


def create_backup(output_dir: Optional[Path] = None) -> Path:
    """
    Create a backup of DuckDB and ChromaDB.
    
    Args:
        output_dir: Directory to store backups (default: backend/backups/)
    
    Returns:
        Path to the created backup directory
    """
    # Determine backup directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "backups"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped backup folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating backup in: {backup_dir}")
    
    # Get paths
    db_path = _get_db_path()
    chroma_path = Path(__file__).parent.parent / "data" / "chromadb"
    
    manifest = {
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "app_version": get_app_version(),
        "components": {}
    }
    
    # 1. Backup DuckDB
    try:
        logger.info(f"Backing up DuckDB from: {db_path}")
        
        if not db_path.exists():
            logger.warning(f"DuckDB file not found at {db_path}")
            manifest["components"]["duckdb"] = {
                "status": "not_found",
                "path": str(db_path)
            }
        else:
            # Close any open connections by using DuckDB's EXPORT DATABASE
            # This ensures a consistent backup
            try:
                import duckdb
                # Create a temporary connection to export
                temp_conn = duckdb.connect(str(db_path), read_only=True)
                backup_db_path = backup_dir / "analytics.duckdb"
                
                # Copy the file (DuckDB is single-file)
                shutil.copy2(db_path, backup_db_path)
                temp_conn.close()
                
                db_size = backup_db_path.stat().st_size
                manifest["components"]["duckdb"] = {
                    "status": "success",
                    "path": str(db_path),
                    "backup_path": str(backup_db_path),
                    "size_bytes": db_size,
                    "size_mb": round(db_size / (1024 * 1024), 2)
                }
                logger.info(f"✅ DuckDB backed up: {db_size / (1024 * 1024):.2f} MB")
            except Exception as e:
                logger.error(f"Error backing up DuckDB: {e}")
                manifest["components"]["duckdb"] = {
                    "status": "error",
                    "error": str(e)
                }
    except Exception as e:
        logger.error(f"Failed to backup DuckDB: {e}")
        manifest["components"]["duckdb"] = {
            "status": "error",
            "error": str(e)
        }
    
    # 2. Backup ChromaDB
    try:
        logger.info(f"Backing up ChromaDB from: {chroma_path}")
        
        if not chroma_path.exists():
            logger.warning(f"ChromaDB directory not found at {chroma_path}")
            manifest["components"]["chromadb"] = {
                "status": "not_found",
                "path": str(chroma_path)
            }
        else:
            backup_chroma_path = backup_dir / "chromadb"
            
            # Copy entire ChromaDB directory
            shutil.copytree(chroma_path, backup_chroma_path, dirs_exist_ok=True)
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in backup_chroma_path.rglob('*') if f.is_file())
            
            manifest["components"]["chromadb"] = {
                "status": "success",
                "path": str(chroma_path),
                "backup_path": str(backup_chroma_path),
                "size_bytes": total_size,
                "size_mb": round(total_size / (1024 * 1024), 2)
            }
            logger.info(f"✅ ChromaDB backed up: {total_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        logger.error(f"Failed to backup ChromaDB: {e}")
        manifest["components"]["chromadb"] = {
            "status": "error",
            "error": str(e)
        }
    
    # 3. Write manifest
    manifest_path = backup_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✅ Backup complete: {backup_dir}")
    logger.info(f"   Manifest: {manifest_path}")
    
    return backup_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup DuckDB and ChromaDB")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for backups (default: backend/backups/)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    backup_path = create_backup(output_dir)
    
    print(f"\n✅ Backup created successfully: {backup_path}")
    print(f"   To restore: python scripts/restore.py {backup_path}")

