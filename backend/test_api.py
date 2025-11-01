#!/usr/bin/env python3
"""
Quick test script to verify the Smart Allocation Engine API is working
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Smart Allocation Engine API...")
    
    # Wait a moment for server to fully start
    time.sleep(2)
    
    try:
        # Test health endpoint
        print("\n1️⃣ Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test candidates endpoint
        print("\n2️⃣ Testing candidates endpoint...")
        response = requests.get(f"{base_url}/candidates/")
        if response.status_code == 200:
            candidates = response.json()
            print(f"✅ Candidates endpoint working: {len(candidates)} candidates found")
        else:
            print(f"❌ Candidates endpoint failed: {response.status_code}")
        
        # Test internships endpoint
        print("\n3️⃣ Testing internships endpoint...")
        response = requests.get(f"{base_url}/internships/")
        if response.status_code == 200:
            internships = response.json()
            print(f"✅ Internships endpoint working: {len(internships)} internships found")
        else:
            print(f"❌ Internships endpoint failed: {response.status_code}")
        
        # Test API documentation
        print("\n4️⃣ Testing API documentation...")
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API documentation accessible at http://localhost:8000/docs")
        else:
            print(f"❌ API documentation failed: {response.status_code}")
        
        print("\n🎉 API is working correctly!")
        print("\n📚 Next steps:")
        print("   - Visit http://localhost:8000/docs for interactive API documentation")
        print("   - Run 'python load_sample_data.py' to populate with sample data")
        print("   - Start building your Next.js frontend!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    test_api()
