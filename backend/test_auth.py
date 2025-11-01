#!/usr/bin/env python3
"""
Test script for authentication endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_auth():
    """Test authentication endpoints"""
    
    print("Testing Authentication System")
    print("=" * 40)
    
    # Test login with admin credentials
    print("\n1. Testing admin login...")
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        token_data = response.json()
        admin_token = token_data["access_token"]
        print(f"✓ Admin login successful. Token: {admin_token[:20]}...")
    else:
        print(f"✗ Admin login failed: {response.status_code} - {response.text}")
        return
    
    # Test login with user credentials
    print("\n2. Testing user login...")
    login_data = {
        "username": "user",
        "password": "user123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        token_data = response.json()
        user_token = token_data["access_token"]
        print(f"✓ User login successful. Token: {user_token[:20]}...")
    else:
        print(f"✗ User login failed: {response.status_code} - {response.text}")
        return
    
    # Test getting current user info
    print("\n3. Testing get current user info...")
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    if response.status_code == 200:
        user_info = response.json()
        print(f"✓ Current user info: {user_info['username']} ({user_info['role']})")
    else:
        print(f"✗ Get user info failed: {response.status_code} - {response.text}")
    
    # Test accessing protected endpoint with admin token
    print("\n4. Testing admin access to candidates...")
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = requests.get(f"{BASE_URL}/candidates/", headers=headers)
    if response.status_code == 200:
        candidates = response.json()
        print(f"✓ Admin can access candidates: {len(candidates)} candidates found")
    else:
        print(f"✗ Admin access to candidates failed: {response.status_code} - {response.text}")
    
    # Test accessing protected endpoint with user token
    print("\n5. Testing user access to candidates...")
    headers = {"Authorization": f"Bearer {user_token}"}
    response = requests.get(f"{BASE_URL}/candidates/", headers=headers)
    if response.status_code == 200:
        candidates = response.json()
        print(f"✓ User can access candidates: {len(candidates)} candidates found")
    else:
        print(f"✗ User access to candidates failed: {response.status_code} - {response.text}")
    
    # Test accessing admin-only endpoint with user token (should fail)
    print("\n6. Testing user access to admin-only endpoint...")
    headers = {"Authorization": f"Bearer {user_token}"}
    response = requests.get(f"{BASE_URL}/auth/users", headers=headers)
    if response.status_code == 403:
        print("✓ User correctly denied access to admin-only endpoint")
    else:
        print(f"✗ User should not have access to admin endpoint: {response.status_code} - {response.text}")
    
    # Test accessing endpoint without token (should fail)
    print("\n7. Testing access without token...")
    response = requests.get(f"{BASE_URL}/candidates/")
    if response.status_code == 401:
        print("✓ Correctly denied access without token")
    else:
        print(f"✗ Should require authentication: {response.status_code} - {response.text}")
    
    print("\n" + "=" * 40)
    print("Authentication test completed!")

if __name__ == "__main__":
    try:
        test_auth()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Make sure the FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")
