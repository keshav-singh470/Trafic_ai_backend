import requests
import json

BASE_URL = "http://localhost:8000"

def test_analytics():
    print("Testing /traffic-analytics...")
    try:
        response = requests.get(f"{BASE_URL}/traffic-analytics")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to connect: {e}")

def test_search(plate):
    print(f"Testing /search-plate?number={plate}...")
    try:
        response = requests.get(f"{BASE_URL}/search-plate", params={"number": plate})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Results found: {len(response.json())}")
            if len(response.json()) > 0:
                print(json.dumps(response.json()[0], indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_analytics()
    # Replace with a real plate from your database if testing live
    test_search("MH12DE1433")
