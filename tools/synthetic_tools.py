import json
import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SETTLEMENTS_FILE = os.path.join(DATA_DIR, "synthetic_settlements.json")
PAYMENTS_FILE = os.path.join(DATA_DIR, "synthetic_payments.json")

def _load_json(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Strip _anomaly internal eval metadata before returning to tools
        cleaned = []
        for item in data:
            item_copy = dict(item)
            item_copy.pop("_anomaly", None)
            cleaned.append(item_copy)
        return cleaned

@tool
def get_synthetic_settlements(
    settlement_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10
) -> str:
    """Fetch settlement records from synthetic dataset. Can filter by settlement_id (setl_XXXX) or status."""
    settlements = _load_json(SETTLEMENTS_FILE)
    if settlement_id:
        filtered = [s for s in settlements if s.get("id") == settlement_id]
    elif status:
        filtered = [s for s in settlements if s.get("status") == status][:limit]
    else:
        filtered = settlements[:limit]
    return json.dumps(filtered, indent=2)

@tool
def get_synthetic_payments(
    payment_id: Optional[str] = None,
    settlement_id: Optional[str] = None,
    limit: int = 10
) -> str:
    """Fetch payment transaction records from synthetic dataset. Can filter by payment_id (pay_XXXX) or settlement_id."""
    payments = _load_json(PAYMENTS_FILE)
    if payment_id:
        filtered = [p for p in payments if p.get("id") == payment_id]
    elif settlement_id:
        filtered = [p for p in payments if p.get("settlement_id") == settlement_id][:limit]
    else:
        filtered = payments[:limit]
    return json.dumps(filtered, indent=2)

@tool
def get_synthetic_settlement_recon(settlement_id: str) -> str:
    """Fetch reconciliation and combined summary report for a specific settlement ID."""
    settlements = _load_json(SETTLEMENTS_FILE)
    payments = _load_json(PAYMENTS_FILE)
    
    target_settlement = next((s for s in settlements if s.get("id") == settlement_id), None)
    if not target_settlement:
        return json.dumps({"error": f"Settlement {settlement_id} not found."})
    
    linked_payments = [p for p in payments if p.get("settlement_id") == settlement_id or p.get("id") in target_settlement.get("payment_ids", [])]
    
    total_payment_amount = sum(p.get("amount", 0) for p in linked_payments)
    total_fees = sum(p.get("fee", 0) for p in linked_payments)
    total_tax = sum(p.get("tax", 0) for p in linked_payments)
    
    recon_report = {
        "settlement": target_settlement,
        "linked_payments_count": len(linked_payments),
        "calculated_total_payment_amount": total_payment_amount,
        "calculated_total_fees": total_fees,
        "calculated_total_tax": total_tax,
        "discrepancy_amount": target_settlement.get("amount", 0) - (total_payment_amount - total_fees - total_tax),
        "linked_payments": linked_payments
    }
    return json.dumps(recon_report, indent=2)
