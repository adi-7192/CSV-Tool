"""
Unit tests for email templates.

Tests that templates render correctly and include required content.
"""
import pytest
from services.email_templates import get_password_reset_template, get_password_changed_template
from core.config import settings


def test_password_reset_template_includes_link():
    """Test that password reset template includes the reset link."""
    reset_url = "https://example.com/reset-password?token=test123"
    expires_minutes = 15
    
    html_body, text_body = get_password_reset_template(reset_url, expires_minutes)
    
    # Check HTML includes link
    assert reset_url in html_body
    assert 'href="' + reset_url + '"' in html_body or f'href="{reset_url}"' in html_body
    assert "Reset Password" in html_body
    
    # Check text includes link
    assert reset_url in text_body
    assert "Reset Your Password" in text_body


def test_password_reset_template_includes_expiry():
    """Test that password reset template includes expiration messaging."""
    reset_url = "https://example.com/reset-password?token=test123"
    expires_minutes = 15
    
    html_body, text_body = get_password_reset_template(reset_url, expires_minutes)
    
    # Check HTML includes expiry
    assert str(expires_minutes) in html_body
    assert "expire" in html_body.lower() or "expires" in html_body.lower()
    
    # Check text includes expiry
    assert str(expires_minutes) in text_body
    assert "expire" in text_body.lower() or "expires" in text_body.lower()


def test_password_reset_template_safety_message():
    """Test that password reset template includes safety message."""
    reset_url = "https://example.com/reset-password?token=test123"
    
    html_body, text_body = get_password_reset_template(reset_url)
    
    # Check safety message
    assert "didn't request" in html_body.lower() or "didn't request" in text_body.lower()
    assert "ignore" in html_body.lower() or "ignore" in text_body.lower()


def test_password_changed_template_includes_login_link():
    """Test that password changed template includes login link."""
    html_body, text_body = get_password_changed_template()
    
    # Check HTML includes login link
    assert "/login" in html_body
    assert "Go to Login" in html_body or "login" in html_body.lower()
    
    # Check text includes login link
    assert "/login" in text_body
    assert "login" in text_body.lower()


def test_password_changed_template_security_warning():
    """Test that password changed template includes security warning."""
    html_body, text_body = get_password_changed_template()
    
    # Check security warning
    assert "didn't make this change" in html_body.lower() or "didn't make this change" in text_body.lower()
    assert "compromised" in html_body.lower() or "compromised" in text_body.lower()


def test_templates_return_both_formats():
    """Test that templates return both HTML and text versions."""
    reset_url = "https://example.com/reset-password?token=test123"
    
    html_body, text_body = get_password_reset_template(reset_url)
    
    assert isinstance(html_body, str)
    assert isinstance(text_body, str)
    assert len(html_body) > 0
    assert len(text_body) > 0
    
    # HTML should be longer (has styling)
    assert len(html_body) > len(text_body)
    
    # Both should contain the reset URL
    assert reset_url in html_body
    assert reset_url in text_body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

