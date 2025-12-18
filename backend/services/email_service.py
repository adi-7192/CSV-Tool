"""
Email Service - Send transactional emails using SMTP.

Supports async sending via FastAPI BackgroundTasks for non-blocking operation.
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

from core.config import settings

logger = logging.getLogger(__name__)


def is_email_configured() -> bool:
    """Check if SMTP settings are configured."""
    return bool(
        settings.SMTP_HOST and
        settings.SMTP_USER and
        settings.SMTP_PASSWORD and
        settings.SMTP_FROM_EMAIL
    )


def send_email(
    to_email: str,
    subject: str,
    html_body: str,
    text_body: Optional[str] = None
) -> bool:
    """
    Send an email via SMTP.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        html_body: HTML content of the email
        text_body: Plain text fallback (optional)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    if not is_email_configured():
        logger.warning(
            "Email not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, "
            "and SMTP_FROM_EMAIL environment variables."
        )
        # In development, log the email content instead
        logger.info(f"[DEV] Would send email to: {to_email}")
        logger.info(f"[DEV] Subject: {subject}")
        logger.info(f"[DEV] Body preview: {html_body[:200]}...")
        return True  # Return True for development flow
    
    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
        message["To"] = to_email
        
        # Add plain text part
        if text_body:
            text_part = MIMEText(text_body, "plain")
            message.attach(text_part)
        
        # Add HTML part
        html_part = MIMEText(html_body, "html")
        message.attach(html_part)
        
        # Create SSL context
        context = ssl.create_default_context()
        
        # Connect and send
        if settings.SMTP_USE_SSL:
            # SSL connection (port 465)
            with smtplib.SMTP_SSL(
                settings.SMTP_HOST,
                settings.SMTP_PORT,
                context=context
            ) as server:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(
                    settings.SMTP_FROM_EMAIL,
                    to_email,
                    message.as_string()
                )
        else:
            # TLS connection (port 587)
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                if settings.SMTP_USE_TLS:
                    server.starttls(context=context)
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(
                    settings.SMTP_FROM_EMAIL,
                    to_email,
                    message.as_string()
                )
        
        logger.info(f"Email sent successfully to: {to_email}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {e}")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending email: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}", exc_info=True)
        return False


def send_password_reset_email(
    to_email: str,
    reset_token: str,
    expires_minutes: int = 15
) -> bool:
    """
    Send password reset email.
    
    Args:
        to_email: User's email address
        reset_token: The reset token (not hashed)
        expires_minutes: Token expiry time in minutes
        
    Returns:
        True if email sent successfully
    """
    reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
    
    subject = "Reset Your Password - Datadost Analytics"
    
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
            <p style="margin: 0;">© {__import__('datetime').datetime.now().year} Datadost Analytics. All rights reserved.</p>
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

© {__import__('datetime').datetime.now().year} Datadost Analytics. All rights reserved.
    """
    
    return send_email(to_email, subject, html_body, text_body)


def send_password_changed_email(to_email: str) -> bool:
    """
    Send notification that password was changed.
    
    Args:
        to_email: User's email address
        
    Returns:
        True if email sent successfully
    """
    subject = "Your Password Has Been Changed - Datadost Analytics"
    
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
                <a href="{settings.FRONTEND_URL}/login" 
                   style="display: inline-block; background: #6366F1; color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px;">
                    Go to Login
                </a>
            </div>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #9ca3af; font-size: 12px;">
            <p style="margin: 0;">© {__import__('datetime').datetime.now().year} Datadost Analytics. All rights reserved.</p>
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

Login: {settings.FRONTEND_URL}/login

© {__import__('datetime').datetime.now().year} Datadost Analytics. All rights reserved.
    """
    
    return send_email(to_email, subject, html_body, text_body)

