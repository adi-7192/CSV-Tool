"""
Tenant Isolation Migration Script

This script performs a one-time migration to enable tenant isolation:
1. Ensures tenant_id column exists in sales table
2. Updates all users to have tenant_id = their user ID
3. Assigns all existing sales data to the admin user (adityadav7192@gmail.com)

Run this script once to migrate existing data:
    cd backend
    python -m scripts.migrate_tenant_isolation

After migration:
- Admin user owns all existing data
- New users will have empty dashboards until they upload their own data
- All future uploads are automatically scoped to the uploading user's tenant_id
"""
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from core.database import get_connection, init_database, table_exists
from services.user_service import get_user_by_email
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Admin email - owner of existing data
ADMIN_EMAIL = "adityadav7192@gmail.com"


def migrate_tenant_isolation():
    """
    Perform tenant isolation migration.
    
    Steps:
    1. Initialize database (ensures tables exist)
    2. Add tenant_id column to sales table if missing
    3. Update all users to have tenant_id = their ID
    4. Assign all existing sales data to admin user
    """
    logger.info("="*60)
    logger.info("TENANT ISOLATION MIGRATION")
    logger.info("="*60)
    
    try:
        # Step 0: Initialize database
        init_database()
        conn = get_connection()
        
        # Step 1: Ensure tenant_id column exists in sales table
        logger.info("\n📊 Step 1: Checking sales table structure...")
        
        if not table_exists('sales'):
            logger.warning("Sales table does not exist - will be created on first upload")
        else:
            # Check if tenant_id column exists
            table_info = conn.execute("DESCRIBE sales").fetchdf()
            columns = table_info['column_name'].tolist()
            
            if 'tenant_id' not in columns:
                logger.info("Adding tenant_id column to sales table...")
                conn.execute("ALTER TABLE sales ADD COLUMN tenant_id VARCHAR")
                logger.info("✅ Added tenant_id column to sales table")
            else:
                logger.info("✅ tenant_id column already exists in sales table")
        
        # Step 2: Update all users to have tenant_id = their ID
        logger.info("\n👤 Step 2: Updating user tenant_ids...")
        
        users_df = conn.execute("SELECT id, email, tenant_id FROM users").fetchdf()
        updated_users = 0
        
        for _, row in users_df.iterrows():
            user_id = row['id']
            email = row['email']
            current_tenant_id = row['tenant_id']
            expected_tenant_id = str(user_id)
            
            if current_tenant_id != expected_tenant_id:
                conn.execute(
                    "UPDATE users SET tenant_id = ? WHERE id = ?",
                    [expected_tenant_id, user_id]
                )
                logger.info(f"  Updated user {email}: tenant_id = {expected_tenant_id}")
                updated_users += 1
            else:
                logger.info(f"  User {email}: tenant_id already correct ({current_tenant_id})")
        
        logger.info(f"✅ Updated {updated_users} users")
        
        # Step 3: Assign all existing sales data to admin user
        logger.info("\n📦 Step 3: Assigning existing sales data to admin...")
        
        # Find admin user
        admin_user = get_user_by_email(ADMIN_EMAIL)
        
        if not admin_user:
            logger.error(f"Admin user {ADMIN_EMAIL} not found!")
            logger.error("Please run 'python -m scripts.seed_admin' first")
            return False
        
        admin_tenant_id = admin_user.tenant_id or str(admin_user.id)
        logger.info(f"Admin user found: ID={admin_user.id}, tenant_id={admin_tenant_id}")
        
        if table_exists('sales'):
            # Count existing sales without tenant_id
            null_tenant_count = conn.execute(
                "SELECT COUNT(*) as count FROM sales WHERE tenant_id IS NULL"
            ).fetchdf().iloc[0]['count']
            
            total_sales = conn.execute(
                "SELECT COUNT(*) as count FROM sales"
            ).fetchdf().iloc[0]['count']
            
            logger.info(f"Total sales records: {total_sales}")
            logger.info(f"Records without tenant_id: {null_tenant_count}")
            
            # Update all sales to admin's tenant_id (existing data belongs to admin)
            if total_sales > 0:
                conn.execute(
                    "UPDATE sales SET tenant_id = ? WHERE tenant_id IS NULL OR tenant_id = ''",
                    [admin_tenant_id]
                )
                logger.info(f"✅ Assigned all existing sales data to admin (tenant_id={admin_tenant_id})")
                
                # Verify update
                admin_sales = conn.execute(
                    "SELECT COUNT(*) as count FROM sales WHERE tenant_id = ?",
                    [admin_tenant_id]
                ).fetchdf().iloc[0]['count']
                logger.info(f"Admin now has {admin_sales} sales records")
        else:
            logger.info("No sales table - skipping sales migration")
        
        # Commit all changes
        conn.commit()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("MIGRATION COMPLETE")
        logger.info("="*60)
        logger.info(f"✅ Admin user: {ADMIN_EMAIL} (tenant_id={admin_tenant_id})")
        logger.info(f"✅ Users updated: {updated_users}")
        logger.info("✅ All existing data assigned to admin")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. Other users will see empty dashboards until they upload data")
        logger.info("2. All new uploads will be scoped to the uploading user")
        logger.info("3. Users cannot see each other's data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = migrate_tenant_isolation()
    sys.exit(0 if success else 1)

