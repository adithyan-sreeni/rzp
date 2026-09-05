import razorpay, os
from dotenv import load_dotenv
load_dotenv()
client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))

def calculate_gross(settlement_id):
    settlement = client.settlement.fetch(settlement_id)
    created_at = settlement["created_at"]
    target_amount = settlement["amount"]
    
    # Try fetching payments prior to this settlement
    payments = client.payment.all({"count": 100})["items"]
    
    # We want to find a subset of payments where sum(p["amount"] - p.get("fee", 0)) == target_amount
    # For a simple greedy approach (assuming they are in order)
    total_gross = 0
    total_fee = 0
    total_tax = 0
    matched_payments = []
    
    # Just sum the most recent payments before created_at until we hit the target amount
    for p in reversed(payments):
        if p["created_at"] <= created_at and p["status"] == "captured":
            net = p["amount"] - p.get("fee", 0)
            if target_amount >= net:
                target_amount -= net
                total_gross += p["amount"]
                total_fee += p.get("fee", 0)
                total_tax += p.get("tax", 0)
                matched_payments.append(p["id"])
            if target_amount == 0:
                break
                
    return {
        "settlement_id": settlement_id,
        "gross_amount": total_gross,
        "total_fee": total_fee,
        "total_tax": total_tax,
        "net_amount": settlement["amount"],
        "matched_payments": matched_payments
    }
    
print(calculate_gross("setl_SpxMT79UCX9FKg"))
