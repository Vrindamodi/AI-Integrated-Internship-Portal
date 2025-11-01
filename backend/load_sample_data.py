#!/usr/bin/env python3
"""
Sample data loader for Smart Allocation Engine
Run this script to populate the database with sample data for testing
"""

import requests
import json
from sample_data import candidates_sample, internships_sample, matching_request_sample

API_BASE_URL = "http://localhost:8000"

# Default admin credentials (from init_auth.py)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def authenticate():
    """Authenticate with the API and return access token"""
    try:
        login_data = {
            "username": ADMIN_USERNAME,
            "password": ADMIN_PASSWORD
        }
        response = requests.post(f"{API_BASE_URL}/auth/login", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            return token_data["access_token"]
        else:
            print(f"❌ Authentication failed: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
        return None

def get_auth_headers(token):
    """Get authentication headers for API requests"""
    return {"Authorization": f"Bearer {token}"}

def load_sample_data():
    """Load sample candidates and internships into the database"""
    
    print("🚀 Loading sample data into Smart Allocation Engine...")
    
    # Authenticate first
    print("🔐 Authenticating...")
    token = authenticate()
    if not token:
        print("❌ Authentication failed. Cannot proceed.")
        return False
    
    print("✅ Authentication successful!")
    headers = get_auth_headers(token)
    
    # Load candidates
    print("\n📝 Loading candidates...")
    for candidate in candidates_sample:
        try:
            response = requests.post(f"{API_BASE_URL}/candidates/", json=candidate, headers=headers)
            if response.status_code == 200:
                print(f"✅ Loaded candidate: {candidate['name']}")
            else:
                print(f"❌ Failed to load candidate {candidate['name']}: {response.text}")
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
            return False
    
    # Load internships
    print("\n🏢 Loading internships...")
    for internship in internships_sample:
        try:
            response = requests.post(f"{API_BASE_URL}/internships/", json=internship, headers=headers)
            if response.status_code == 200:
                print(f"✅ Loaded internship: {internship['role']} at {internship['org']}")
            else:
                print(f"❌ Failed to load internship {internship['role']}: {response.text}")
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
            return False
    
    print("\n🎉 Sample data loaded successfully!")
    return True

def run_sample_matching():
    """Run the matching algorithm with sample data"""
    
    print("\n🔄 Running matching algorithm...")
    
    # Authenticate first
    token = authenticate()
    if not token:
        print("❌ Authentication failed. Cannot run matching.")
        return
    
    headers = get_auth_headers(token)
    
    try:
        response = requests.post(f"{API_BASE_URL}/matching/run", json=matching_request_sample, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Matching completed successfully!")
            print(f"   - Total candidates: {result['total_candidates']}")
            print(f"   - Total internships: {result['total_internships']}")
            print(f"   - Matched: {result['matched_count']}")
            print(f"   - Waitlist: {result['waitlist_count']}")
            print(f"   - Message: {result['message']}")
            
            # Show some placements
            if result['placements']:
                print("\n📋 Sample placements:")
                for placement in result['placements'][:3]:  # Show first 3
                    print(f"   - {placement['candidate_id']} → {placement['internship_id']} (Score: {placement['total_score']:.3f})")
        else:
            print(f"❌ Matching failed: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")

def show_api_info():
    """Show API information and available endpoints"""
    
    print("\n📚 API Information:")
    print(f"   - API Documentation: {API_BASE_URL}/docs")
    print(f"   - Alternative Docs: {API_BASE_URL}/redoc")
    print(f"   - Health Check: {API_BASE_URL}/health")
    
    print("\n🔗 Key Endpoints:")
    print("   - GET /candidates/ - List all candidates")
    print("   - GET /internships/ - List all internships")
    print("   - GET /placements/ - List all placements")
    print("   - POST /matching/run - Run allocation algorithm")
    print("   - POST /waitlist/promote - Promote from waitlist")

def main():
    """Main function"""
    
    print("🎯 Smart Allocation Engine - Sample Data Loader")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("❌ API server is not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running!")
        print("Please start the server first:")
        print("   uvicorn app:app --reload")
        return
    
    # Check if authentication is set up
    print("\n🔐 Checking authentication setup...")
    try:
        # Try to authenticate to check if admin user exists
        token = authenticate()
        if token:
            print("✅ Authentication working - admin user exists")
        else:
            print("❌ Authentication failed - admin user may not exist")
            print("Please run: python init_auth.py")
            return
    except Exception as e:
        print(f"❌ Authentication check failed: {e}")
        print("Please run: python init_auth.py")
        return
    
    # Load sample data
    if load_sample_data():
        # Run sample matching
        run_sample_matching()
        
        # Show API info
        show_api_info()
        
        print("\n🎉 Setup complete! You can now:")
        print("   1. Visit http://localhost:8000/docs to explore the API")
        print("   2. Use the frontend to interact with the system")
        print("   3. Test different matching scenarios")
        print("\n🔑 Authentication Info:")
        print(f"   - Admin: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
        print("   - User: user / user123")
        print("   - ⚠️  Change these passwords in production!")

if __name__ == "__main__":
    main()
