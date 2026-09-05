import os
import json
import time
import asyncio
from typing import Dict, Any, List
from graph.agent_graph import build_graph
from eval.scorer import keyword_score, semantic_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GROUND_TRUTH_FILE = os.path.join(DATA_DIR, "ground_truth_qa.json")

async def run_evaluation():
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Ground truth file not found at {GROUND_TRUTH_FILE}. Please run scripts/generate_synthetic_data.py first.")
        return

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    graph = build_graph()
    
    keyword_passes = 0
    semantic_passes = 0
    total_questions = len(qa_pairs)
    failed_cases = []
    exceptions_list = []
    latencies = []

    print(f"Starting eval harness on {total_questions} ground truth Q&A pairs...\n")

    for item in qa_pairs:
        q_id = item["id"]
        question = item["question"]
        expected_keywords = item["expected_answer_keywords"]
        
        start_t = time.time()
        try:
            result = await graph.ainvoke({"question": question})
            latency = time.time() - start_t
            latencies.append(latency)
            
            final_answer = result.get("final_answer", "")
            exceptions = result.get("exceptions", [])
            if exceptions:
                exceptions_list.append((q_id, exceptions))
                
            kw_pass = keyword_score(final_answer, expected_keywords)
            sem_pass = semantic_score(final_answer, expected_keywords)
            
            if kw_pass:
                keyword_passes += 1
            if sem_pass:
                semantic_passes += 1
            if not kw_pass and not sem_pass:
                failed_cases.append({
                    "id": q_id,
                    "question": question,
                    "expected_keywords": expected_keywords,
                    "received_answer": final_answer
                })
        except Exception as e:
            latencies.append(time.time() - start_t)
            exceptions_list.append((q_id, [str(e)]))
            failed_cases.append({
                "id": q_id,
                "question": question,
                "expected_keywords": expected_keywords,
                "received_answer": f"Error: {e}"
            })

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    kw_pct = (keyword_passes / total_questions) * 100 if total_questions else 0
    sem_pct = (semantic_passes / total_questions) * 100 if total_questions else 0

    print("=== EVAL RESULTS ===")
    print(f"Total questions: {total_questions}")
    print(f"Passed (keyword): {keyword_passes}/{total_questions} ({kw_pct:.1f}%)")
    print(f"Passed (semantic): {semantic_passes}/{total_questions} ({sem_pct:.1f}%)")
    print(f"Average latency: {avg_latency:.2f}s per question")
    
    if failed_cases:
        print("\nFailed cases:")
        for fc in failed_cases:
            print(f"  - {fc['id']}: Expected '{fc['expected_keywords']}' — Agent answer preview: '{fc['received_answer'][:100]}...'")
            
    if exceptions_list:
        print("\nException list:")
        for q_id, excs in exceptions_list:
            print(f"  - {q_id}: {', '.join(excs)}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
