#!/usr/bin/env python3
# Quick test for Salesforce tools (token + get_all_orders).
# Run from backend:  PYTHONPATH=. python scripts/test_salesforce_tools.py
import sys
from pathlib import Path

backend = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend))

def main():
    from app.tools.salesforce_tool import salesforce_get_all_orders

    print("Testing Salesforce: token + get_all_orders...")
    result = salesforce_get_all_orders.invoke({})
    if result.get("error"):
        print("FAIL:", result["error"])
        print("Check .env: SF_TOKEN_URL, SF_CLIENT_ID, SF_CLIENT_SECRET, SF_API_BASE_URL")
        print("SF_API_BASE_URL must end with /services/data/v60.0")
        return 1
    n = result.get("totalSize", 0)
    records = result.get("records", [])
    print(f"OK: totalSize={n}, records={len(records)}")
    for i, rec in enumerate(records[:3]):
        print(f"  [{i+1}] OrderNumber={rec.get('OrderNumber')} Status={rec.get('Status')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
