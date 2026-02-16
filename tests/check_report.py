import requests
import pandas as pd
import json

JOB_ID = "0908b2fd-8dd1-405d-91bf-339e4506fe2c"
API_URL = "http://localhost:8000"

def check_report():
    try:
        res = requests.get(f"{API_URL}/report/{JOB_ID}")
        if res.status_code == 200:
            data = res.json()
            print(f"Total violations detected so far: {len(data)}")
            if data:
                print("First 5 violations:")
                print(json.dumps(data[:5], indent=2))
                
                # Save partial report
                df = pd.DataFrame(data)
                df.to_csv("helmet_partial_report.csv", index=False)
                print("Partial report saved to helmet_partial_report.csv")
        else:
            print(f"Error fetching report: {res.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_report()
