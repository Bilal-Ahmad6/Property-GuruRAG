#!/usr/bin/env python3
"""
Simple test script to verify the FastAPI app works with lazy loading.
"""

import requests
import json
import time

def test_health_endpoint():
    """Test the health endpoint for fast response."""
    print("Testing /health endpoint...")
    start = time.time()
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        elapsed = (time.time() - start) * 1000
        
        print(f"✓ Health check responded in {elapsed:.2f}ms")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint."""
    print("\nTesting / endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"✓ Root endpoint responded")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False

def test_query_endpoint():
    """Test the query endpoint (will fail if no data, but should not crash)."""
    print("\nTesting /query endpoint...")
    
    payload = {
        "question": "What properties are available in Bahria Town?",
        "k": 3
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/query", 
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Query endpoint responded successfully")
            print(f"  Processing time: {result.get('processing_time_ms', 'N/A')}ms")
            print(f"  Results found: {len(result.get('results', []))}")
        else:
            print(f"⚠ Query endpoint returned {response.status_code}")
            print(f"  Error: {response.text}")
        
        return True
    except Exception as e:
        print(f"✗ Query endpoint failed: {e}")
        return False

def test_collections_endpoint():
    """Test the collections listing endpoint."""
    print("\nTesting /collections endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/collections", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Collections endpoint responded successfully")
            print(f"  Collections: {result.get('collections', [])}")
            print(f"  Count: {result.get('count', 0)}")
        else:
            print(f"⚠ Collections endpoint returned {response.status_code}")
            print(f"  Error: {response.text}")
        
        return True
    except Exception as e:
        print(f"✗ Collections endpoint failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing FastAPI PropertyGuru RAG API")
    print("=" * 50)
    
    # Test endpoints
    health_ok = test_health_endpoint()
    root_ok = test_root_endpoint()
    query_ok = test_query_endpoint()
    collections_ok = test_collections_endpoint()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Health endpoint: {'✓' if health_ok else '✗'}")
    print(f"Root endpoint: {'✓' if root_ok else '✗'}")
    print(f"Query endpoint: {'✓' if query_ok else '✗'}")
    print(f"Collections endpoint: {'✓' if collections_ok else '✗'}")
    
    if health_ok and root_ok:
        print("\n🎉 Core functionality working! App should deploy successfully to Render.")
    else:
        print("\n❌ Some issues detected. Check the logs above.")
