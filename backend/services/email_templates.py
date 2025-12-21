"""
Email templates for transactional emails.

All templates return both HTML and plain text versions.
"""
from datetime import datetime
from typing import Tuple
from core.config import settings


def get_password_reset_template(
    reset_url: str,
    expires_minutes: int = 15
) -> Tuple[str, str]:
    """
    Generate password reset email template.
    
    Args:
        reset_url: Full URL to reset password page with token
        expires_minutes: Token expiration time in minutes
        
    Returns:
        Tuple of (html_body, text_body)
    """
    current_year = datetime.now().year
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
            <h1 style="color: white; margin: 0; font-size: 24px;">Datadost Analytics</h1>
        </div>
        
        <div style="background: #ffffff; padding: 40px 30px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 10px 10px;">
            <h2 style="color: #1f2937; margin-top: 0;">Reset Your Password</h2>
            
            <p style="color: #4b5563;">
                We received a request to reset the password for your account. 
                Click the button below to create a new password.
            </p>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="{reset_url}" 
                   style="display: inline-block; background: #6366F1; color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px;">
                    Reset Password
                </a>
            </div>
            
            <p style="color: #6b7280; font-size: 14px;">
                This link will expire in <strong>{expires_minutes} minutes</strong>.
            </p>
            
            <p style="color: #6b7280; font-size: 14px;">
                If you didn't request a password reset, you can safely ignore this email. 
                Your password will remain unchanged.
            </p>
            
            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">
            
            <p style="color: #9ca3af; font-size: 12px; margin-bottom: 5px;">
                If the button doesn't work, copy and paste this link into your browser:
            </p>
            <p style="color: #6366F1; font-size: 12px; word-break: break-all;">
                {reset_url}
            </p>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #9ca3af; font-size: 12px;">
            <p style="margin: 0;">© {current_year} Datadost Analytics. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
Reset Your Password - Datadost Analytics

We received a request to reset the password for your account.

Click this link to reset your password:
{reset_url}

This link will expire in {expires_minutes} minutes.

If you didn't request a password reset, you can safely ignore this email.
Your password will remain unchanged.

© {current_year} Datadost Analytics. All rights reserved.
    """
    
    return html_body.strip(), text_body.strip()


def get_password_changed_template() -> Tuple[str, str]:
    """
    Generate password changed notification email template.
    
    Returns:
        Tuple of (html_body, text_body)
    """
    current_year = datetime.now().year
    login_url = f"{settings.FRONTEND_URL}/login"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
            <h1 style="color: white; margin: 0; font-size: 24px;">Datadost Analytics</h1>
        </div>
        
        <div style="background: #ffffff; padding: 40px 30px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 10px 10px;">
            <h2 style="color: #1f2937; margin-top: 0;">Password Changed Successfully</h2>
            
            <p style="color: #4b5563;">
                Your password has been successfully changed. You can now log in with your new password.
            </p>
            
            <div style="background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 16px; margin: 20px 0; border-radius: 4px;">
                <p style="color: #92400E; margin: 0; font-size: 14px;">
                    <strong>Didn't make this change?</strong><br>
                    If you didn't change your password, please contact our support immediately 
                    as your account may have been compromised.
                </p>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="{login_url}" 
                   style="display: inline-block; background: #6366F1; color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px;">
                    Go to Login
                </a>
            </div>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #9ca3af; font-size: 12px;">
            <p style="margin: 0;">© {current_year} Datadost Analytics. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
Password Changed Successfully - Datadost Analytics

Your password has been successfully changed. You can now log in with your new password.

Didn't make this change?
If you didn't change your password, please contact our support immediately 
as your account may have been compromised.

Login: {login_url}

© {current_year} Datadost Analytics. All rights reserved.
    """
    
    return html_body.strip(), text_body.strip()

