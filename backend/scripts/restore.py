#!/usr/bin/env python3
"""
Restore script for DuckDB and ChromaDB

Restores from a backup created by backup.py

Usage:
    python scripts/restore.py <backup_dir> [--confirm]
"""
import sys
import shutil
import json
from pathlib import Path
from typing import Optional
import argparse
import logging

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from core.database import _get_db_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def restore_backup(backup_dir: Path, confirm: bool = False) -> bool:
    """
    Restore DuckDB and ChromaDB from backup.
    
    Args:
        backup_dir: Path to backup directory
        confirm: Whether user has confirmed the restore
    
    Returns:
        True if successful, False otherwise
    """
    backup_dir = Path(backup_dir)
    
    if not backup_dir.exists():
        logger.error(f"Backup directory not found: {backup_dir}")
        return False
    
    # Read manifest
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found in backup: {manifest_path}")
        return False
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    logger.info(f"Restoring from backup: {backup_dir}")
    logger.info(f"  Created: {manifest.get('created_at', 'unknown')}")
    logger.info(f"  Git SHA: {manifest.get('git_sha', 'unknown')}")
    logger.info(f"  App Version: {manifest.get('app_version', 'unknown')}")
    
    if not confirm:
        print("\n⚠️  WARNING: This will overwrite your current database and ChromaDB data!")
        print("⚠️  Make sure the application is STOPPED before restoring.")
        print("\nTo confirm, run with --confirm flag:")
        print(f"  python scripts/restore.py {backup_dir} --confirm")
        return False
    
    # Get target paths
    db_path = _get_db_path()
    chroma_path = Path(__file__).parent.parent / "data" / "chromadb"
    
    # 1. Restore DuckDB
    duckdb_status = manifest.get("components", {}).get("duckdb", {})
    if duckdb_status.get("status") == "success":
        try:
            backup_db_path = backup_dir / "analytics.duckdb"
            if not backup_db_path.exists():
                logger.error(f"DuckDB backup file not found: {backup_db_path}")
            else:
                logger.info(f"Restoring DuckDB to: {db_path}")
                
                # Ensure parent directory exists
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Backup current DB if it exists (safety)
                if db_path.exists():
                    safety_backup = db_path.parent / f"{db_path.name}.pre_restore_backup"
                    logger.info(f"Creating safety backup of current DB: {safety_backup}")
                    shutil.copy2(db_path, safety_backup)
                
                # Copy backup to target
                shutil.copy2(backup_db_path, db_path)
                logger.info(f"✅ DuckDB restored: {db_path.stat().st_size / (1024 * 1024):.2f} MB")
        except Exception as e:
            logger.error(f"Failed to restore DuckDB: {e}")
            return False
    else:
        logger.warning(f"DuckDB backup status: {duckdb_status.get('status', 'unknown')}")
    
    # 2. Restore ChromaDB
    chromadb_status = manifest.get("components", {}).get("chromadb", {})
    if chromadb_status.get("status") == "success":
        try:
            backup_chroma_path = backup_dir / "chromadb"
            if not backup_chroma_path.exists():
                logger.error(f"ChromaDB backup directory not found: {backup_chroma_path}")
            else:
                logger.info(f"Restoring ChromaDB to: {chroma_path}")
                
                # Backup current ChromaDB if it exists (safety)
                if chroma_path.exists():
                    safety_backup = chroma_path.parent / f"{chroma_path.name}.pre_restore_backup"
                    logger.info(f"Creating safety backup of current ChromaDB: {safety_backup}")
                    if safety_backup.exists():
                        shutil.rmtree(safety_backup)
                    shutil.copytree(chroma_path, safety_backup)
                
                # Remove existing ChromaDB
                if chroma_path.exists():
                    shutil.rmtree(chroma_path)
                
                # Copy backup to target
                shutil.copytree(backup_chroma_path, chroma_path)
                logger.info(f"✅ ChromaDB restored")
        except Exception as e:
            logger.error(f"Failed to restore ChromaDB: {e}")
            return False
    else:
        logger.warning(f"ChromaDB backup status: {chromadb_status.get('status', 'unknown')}")
    
    logger.info("✅ Restore complete!")
    logger.info("⚠️  Remember to restart the application after restore.")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore DuckDB and ChromaDB from backup")
    parser.add_argument(
        "backup_dir",
        type=str,
        help="Path to backup directory"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm restore (required to proceed)"
    )
    
    args = parser.parse_args()
    
    backup_dir = Path(args.backup_dir)
    success = restore_backup(backup_dir, confirm=args.confirm)
    
    sys.exit(0 if success else 1)

