import json
import os
import random
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def generate_synthetic_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    random.seed(42)
    base_time = 1725000000  # Epoch time ~ Sept 2024
    
    settlements = []
    payments = []
    qa_pairs = []
    
    utr_pool = [f"HDFC202509{i:07d}" for i in range(1, 70)]
    
    # 80 settlement records
    # 8 unexplained_deduction, 8 timing_mismatch, 4 duplicate_utr, 4 amount_mismatch, 56 clean
    anomaly_types = (
        ["unexplained_deduction"] * 8 +
        ["timing_mismatch"] * 8 +
        ["duplicate_utr"] * 4 +
        ["amount_mismatch"] * 4 +
        ["none"] * 56
    )
    random.shuffle(anomaly_types)
    
    duplicate_utr_val = "HDFC2025099999999"
    
    for i in range(80):
        setl_id = f"setl_{i+1:06d}"
        created_at = base_time + (i * 86400)
        anomaly = anomaly_types[i]
        
        # Base payment values
        num_payments = random.randint(5, 15)
        setl_payments = []
        sum_pay_amount = 0
        sum_fee = 0
        sum_tax = 0
        
        for p_idx in range(num_payments):
            pay_id = f"pay_{i+1:04d}_{p_idx+1:02d}"
            order_id = f"order_{i+1:04d}_{p_idx+1:02d}"
            pay_amount = random.randint(10000, 150000)  # 100 to 1500 INR in paise
            fee = int(pay_amount * 0.02)  # 2% fee
            tax = int(fee * 0.18)        # 18% GST on fee
            
            p_record = {
                "id": pay_id,
                "entity": "payment",
                "amount": pay_amount,
                "currency": "INR",
                "status": "captured",
                "order_id": order_id,
                "method": random.choice(["upi", "card", "netbanking"]),
                "amount_refunded": 0,
                "refund_status": None,
                "captured": True,
                "fee": fee,
                "tax": tax,
                "settlement_id": setl_id,
                "created_at": created_at + random.randint(100, 3600)
            }
            payments.append(p_record)
            setl_payments.append(pay_id)
            sum_pay_amount += pay_amount
            sum_fee += fee
            sum_tax += tax
            
        base_settlement_amount = sum_pay_amount - sum_fee - sum_tax
        fees = sum_fee
        tax = sum_tax
        status = "processed"
        
        if anomaly == "timing_mismatch":
            settled_at = created_at + (7 * 86400)  # 7 days delay (> 5 days)
            utr = utr_pool[i % len(utr_pool)]
            anomaly_data = {
                "type": "timing_mismatch",
                "description": "Settlement delayed by more than 5 days",
                "ground_truth_explanation": "Bank clearance delayed during national holiday"
            }
        elif anomaly == "unexplained_deduction":
            settled_at = created_at + (1 * 86400)
            utr = utr_pool[i % len(utr_pool)]
            fees = int(sum_pay_amount * 0.05)  # 5% fee instead of 2%
            base_settlement_amount = sum_pay_amount - fees - tax
            anomaly_data = {
                "type": "unexplained_deduction",
                "description": "Fee rate of 5% exceeds standard 2% agreement",
                "ground_truth_explanation": "Additional risk surcharge applied"
            }
        elif anomaly == "duplicate_utr":
            settled_at = created_at + (1 * 86400)
            utr = duplicate_utr_val
            anomaly_data = {
                "type": "duplicate_utr",
                "description": f"UTR {duplicate_utr_val} reused across multiple settlements",
                "ground_truth_explanation": "Batch transfer grouping multiple settlements under single UTR"
            }
        elif anomaly == "amount_mismatch":
            settled_at = created_at + (1 * 86400)
            utr = utr_pool[i % len(utr_pool)]
            base_settlement_amount = base_settlement_amount - 50000  # 500 INR mismatch
            anomaly_data = {
                "type": "amount_mismatch",
                "description": "Settlement amount does not match sum of net payments",
                "ground_truth_explanation": "Manual chargeback deduction of ₹500"
            }
        else:
            settled_at = created_at + (1 * 86400)
            utr = utr_pool[i % len(utr_pool)]
            anomaly_data = None

        s_record = {
            "id": setl_id,
            "entity": "settlement",
            "amount": base_settlement_amount,
            "status": status,
            "fees": fees,
            "tax": tax,
            "utr": utr,
            "description": None,
            "created_at": created_at,
            "settled_at": settled_at,
            "payments_count": len(setl_payments),
            "payment_ids": setl_payments
        }
        if anomaly_data:
            s_record["_anomaly"] = anomaly_data
            
        settlements.append(s_record)

    # Generate Ground Truth QA (25 questions)
    qa_categories = [
        "deduction_explanation", "settlement_status", "timing_query",
        "anomaly_detection", "aggregation", "comparison_across_settlements"
    ]
    
    q_counter = 1
    # Find specific settlements for ground truth
    unexplained_setl = [s for s in settlements if s.get("_anomaly", {}).get("type") == "unexplained_deduction"]
    timing_setl = [s for s in settlements if s.get("_anomaly", {}).get("type") == "timing_mismatch"]
    mismatch_setl = [s for s in settlements if s.get("_anomaly", {}).get("type") == "amount_mismatch"]
    duplicate_setl = [s for s in settlements if s.get("_anomaly", {}).get("type") == "duplicate_utr"]
    clean_setl = [s for s in settlements if "_anomaly" not in s]
    
    # 5 deduction_explanation
    for idx in range(5):
        s = unexplained_setl[idx % len(unexplained_setl)]
        qa_pairs.append({
            "id": f"q{q_counter:03d}",
            "question": f"Why are the fees higher for settlement {s['id']}?",
            "expected_answer_keywords": ["fee", "rate", "5%", "deduction", "surcharge"],
            "relevant_record_ids": [s['id']],
            "category": "deduction_explanation",
            "difficulty": "medium"
        })
        q_counter += 1
        
    # 4 settlement_status
    for idx in range(4):
        s = clean_setl[idx]
        qa_pairs.append({
            "id": f"q{q_counter:03d}",
            "question": f"What is the status and UTR number of settlement {s['id']}?",
            "expected_answer_keywords": [s['status'], s['utr']],
            "relevant_record_ids": [s['id']],
            "category": "settlement_status",
            "difficulty": "easy"
        })
        q_counter += 1
        
    # 4 timing_query
    for idx in range(4):
        s = timing_setl[idx % len(timing_setl)]
        qa_pairs.append({
            "id": f"q{q_counter:03d}",
            "question": f"Was settlement {s['id']} delayed beyond the expected processing timeframe?",
            "expected_answer_keywords": ["delayed", "7 days", "timing", "holiday"],
            "relevant_record_ids": [s['id']],
            "category": "timing_query",
            "difficulty": "medium"
        })
        q_counter += 1
        
    # 4 anomaly_detection
    for idx in range(4):
        s = mismatch_setl[idx % len(mismatch_setl)]
        qa_pairs.append({
            "id": f"q{q_counter:03d}",
            "question": f"Does settlement {s['id']} have any discrepancy between the settlement amount and payment net totals?",
            "expected_answer_keywords": ["discrepancy", "mismatch", "chargeback", "500"],
            "relevant_record_ids": [s['id']],
            "category": "anomaly_detection",
            "difficulty": "hard"
        })
        q_counter += 1
        
    # 4 aggregation
    for idx in range(4):
        qa_pairs.append({
            "id": f"q{q_counter:03d}",
            "question": f"How many settlements were processed in the dataset?",
            "expected_answer_keywords": ["80", "settlements"],
            "relevant_record_ids": [],
            "category": "aggregation",
            "difficulty": "easy"
        })
        q_counter += 1
        
    # 4 comparison_across_settlements
    for idx in range(4):
        s1 = clean_setl[idx]
        s2 = clean_setl[idx+5]
        qa_pairs.append({
            "id": f"q{q_counter:03d}",
            "question": f"Which settlement has a higher payout amount between {s1['id']} and {s2['id']}?",
            "expected_answer_keywords": [s1['id'] if s1['amount'] >= s2['amount'] else s2['id']],
            "relevant_record_ids": [s1['id'], s2['id']],
            "category": "comparison_across_settlements",
            "difficulty": "medium"
        })
        q_counter += 1

    # Save to files
    with open(os.path.join(DATA_DIR, "synthetic_settlements.json"), "w", encoding="utf-8") as f:
        json.dump(settlements, f, indent=2)
        
    with open(os.path.join(DATA_DIR, "synthetic_payments.json"), "w", encoding="utf-8") as f:
        json.dump(payments, f, indent=2)
        
    with open(os.path.join(DATA_DIR, "ground_truth_qa.json"), "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Generated {len(settlements)} synthetic settlements, {len(payments)} payments, and {len(qa_pairs)} ground truth Q&A pairs in {DATA_DIR}")

if __name__ == "__main__":
    generate_synthetic_dataset()
