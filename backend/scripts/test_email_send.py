#!/usr/bin/env python3
"""
Test script to send a password reset email and verify it works.

Run this to test if email sending is working correctly.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from core.config import settings
from services.email_service import send_password_reset_email
from services.password_reset_service import create_password_reset_token
from services.user_service import get_user_by_email
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("Password Reset Email Test")
    print("=" * 80)
    
    # Get email from command line or use default
    if len(sys.argv) > 1:
        test_email = sys.argv[1]
    else:
        test_email = input("Enter email address to test: ").strip()
    
    if not test_email:
        print("❌ No email provided")
        return
    
    print(f"\nTesting password reset email for: {test_email}")
    print(f"ENV: {settings.ENV}")
    print(f"SMTP_HOST: {settings.SMTP_HOST}")
    print(f"SMTP_FROM: {settings.SMTP_FROM}")
    print(f"FRONTEND_URL: {settings.FRONTEND_URL}")
    print()
    
    # Check if user exists
    user = get_user_by_email(test_email)
    if not user:
        print(f"⚠️  User with email {test_email} does not exist in database")
        print("Creating a test token anyway...")
        # Create a dummy user ID for testing
        user_id = 999
    else:
        user_id = user.id
        print(f"✓ User found: ID={user_id}")
    
    # Create reset token
    print("\nCreating password reset token...")
    try:
        token = create_password_reset_token(user_id)
        print(f"✓ Token created: {token[:20]}...")
    except Exception as e:
        print(f"❌ Error creating token: {e}")
        return
    
    # Send email
    print("\nSending password reset email...")
    try:
        result = send_password_reset_email(
            to_email=test_email,
            reset_token=token,
            expires_minutes=15
        )
        
        if result:
            print("✓ Email send function returned True")
            print("\n" + "=" * 80)
            print("SUCCESS - Email should have been sent")
            print("=" * 80)
            print(f"\nReset URL: {settings.FRONTEND_URL}/reset-password?token={token}")
            print(f"\n⚠️  Please check:")
            print(f"   1. Inbox for: {test_email}")
            print(f"   2. Spam/Junk folder")
            print(f"   3. Backend logs for any errors")
            print(f"\nIf email is not received, check:")
            print(f"   - Gmail security settings")
            print(f"   - App Password is correct")
            print(f"   - Backend server logs")
        else:
            print("❌ Email send function returned False")
            print("Check backend logs for errors")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

