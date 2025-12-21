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
from typing import Tuple

logger = logging.getLogger(__name__)


def is_email_configured() -> bool:
    """Check if SMTP settings are configured."""
    return bool(
        settings.SMTP_HOST and
        settings.SMTP_USERNAME and
        settings.SMTP_PASSWORD and
        settings.SMTP_FROM
    )


def validate_email_config() -> Tuple[bool, Optional[str]]:
    """
    Validate email configuration based on environment.
    
    Returns:
        Tuple of (is_valid, error_message)
        - In production: returns False with error if SMTP not configured
        - In development: always returns True (allows dry-run mode)
    """
    if settings.ENV.lower() == "production":
        missing = []
        if not settings.SMTP_HOST:
            missing.append("SMTP_HOST")
        if not settings.SMTP_USERNAME:
            missing.append("SMTP_USERNAME")
        if not settings.SMTP_PASSWORD:
            missing.append("SMTP_PASSWORD")
        if not settings.SMTP_FROM:
            missing.append("SMTP_FROM")
        
        if missing:
            error_msg = f"Missing required SMTP configuration in production: {', '.join(missing)}"
            return False, error_msg
    
    return True, None


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
        
    Raises:
        ValueError: In production mode if SMTP is not configured
    """
    # Validate configuration based on environment
    is_valid, error_msg = validate_email_config()
    if not is_valid:
        logger.error(error_msg)
        if settings.ENV.lower() == "production":
            raise ValueError(error_msg)
    
    # Dry-run mode for development
    if not is_email_configured():
        logger.info("=" * 80)
        logger.info("[DEV MODE] Email not configured - DRY RUN (email not sent)")
        logger.info(f"To: {to_email}")
        logger.info(f"Subject: {subject}")
        logger.info("-" * 80)
        if text_body:
            logger.info("Text Body:")
            logger.info(text_body)
            logger.info("-" * 80)
        logger.info("HTML Body Preview:")
        logger.info(html_body[:500] + ("..." if len(html_body) > 500 else ""))
        logger.info("=" * 80)
        return True  # Return True for development flow
    
    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM}>"
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
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.sendmail(
                    settings.SMTP_FROM,
                    to_email,
                    message.as_string()
                )
        else:
            # TLS connection (port 587)
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                if settings.SMTP_USE_TLS:
                    server.starttls(context=context)
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.sendmail(
                    settings.SMTP_FROM,
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
        True if email sent successfully (or dry-run in dev mode)
    """
    from services.email_templates import get_password_reset_template
    from urllib.parse import quote
    
    # Build reset URL with properly encoded token
    # Format: {FRONTEND_URL}/reset-password?token={token}
    reset_url = f"{settings.FRONTEND_URL.rstrip('/')}/reset-password?token={quote(reset_token)}"
    subject = "Reset Your Password - Datadost Analytics"
    
    # Log the reset URL in development for debugging
    if settings.ENV.lower() == "development":
        logger.info(f"[DEV] Sending password reset link: {reset_url}")
    
    html_body, text_body = get_password_reset_template(reset_url, expires_minutes)
    
    try:
        logger.info(f"Attempting to send password reset email to: {to_email}")
        result = send_email(to_email, subject, html_body, text_body)
        if result:
            logger.info(f"Password reset email sent successfully to: {to_email}")
        else:
            logger.error(f"Failed to send password reset email to: {to_email} (send_email returned False)")
        return result
    except ValueError as e:
        # In production, if SMTP is not configured, log error but don't fail the request
        logger.error(f"Failed to send password reset email: {e}", exc_info=True)
        return False
    except Exception as e:
        # Catch any other exceptions and log them
        logger.error(f"Unexpected error sending password reset email to {to_email}: {e}", exc_info=True)
        return False


def send_password_changed_email(to_email: str) -> bool:
    """
    Send notification that password was changed.
    
    Args:
        to_email: User's email address
        
    Returns:
        True if email sent successfully (or dry-run in dev mode)
    """
    from services.email_templates import get_password_changed_template
    
    subject = "Your Password Has Been Changed - Datadost Analytics"
    html_body, text_body = get_password_changed_template()
    
    try:
        return send_email(to_email, subject, html_body, text_body)
    except ValueError as e:
        # In production, if SMTP is not configured, log error but don't fail the request
        logger.error(f"Failed to send password changed email: {e}")
        return False

