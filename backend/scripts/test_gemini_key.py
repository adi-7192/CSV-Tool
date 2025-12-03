#!/usr/bin/env python3
"""
Test Gemini API Key Script

Quick script to test if a Gemini API key is valid.
Run: python backend/scripts/test_gemini_key.py YOUR_API_KEY
"""

import sys
import asyncio
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_gemini_key_quick(api_key: str):
    """Quick test of Gemini API key with minimal timeout"""
    print(f"\n🔍 Testing Gemini API Key...")
    print(f"Key length: {len(api_key)}")
    print(f"Key preview: {api_key[:10]}...{api_key[-4:]}\n")
    
    # Try the most common endpoint first
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": "Hi"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 1
        }
    }
    
    try:
        print(f"📡 Calling: {endpoint.split('?')[0]}")
        print(f"⏱️  Timeout: 10 seconds\n")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            
            print(f"✅ Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API Key is VALID!")
                print(f"Response: {result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'N/A')}")
                return True
            else:
                print(f"❌ API Key validation FAILED")
                print(f"Status: {response.status_code}")
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    print(f"Error: {error_msg}")
                except:
                    print(f"Response: {response.text[:200]}")
                return False
                
    except httpx.TimeoutException:
        print(f"❌ Request TIMED OUT after 10 seconds")
        print(f"This could mean:")
        print(f"  1. Network connectivity issues")
        print(f"  2. Gemini API is slow or unavailable")
        print(f"  3. API key has restrictions blocking the request")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_gemini_key.py YOUR_API_KEY")
        sys.exit(1)
    
    api_key = sys.argv[1].strip()
    
    if len(api_key) < 20:
        print("❌ Error: API key appears too short")
        sys.exit(1)
    
    result = asyncio.run(test_gemini_key_quick(api_key))
    sys.exit(0 if result else 1)

