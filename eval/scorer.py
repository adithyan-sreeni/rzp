import logging
from typing import List

logger = logging.getLogger(__name__)

def keyword_score(answer: str, keywords: List[str]) -> bool:
    if not answer or not keywords:
        return False
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)

def semantic_score(answer: str, expected_keywords: List[str], threshold: float = 0.75) -> bool:
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        expected_text = " ".join(expected_keywords)
        emb1 = model.encode(answer, convert_to_tensor=True)
        emb2 = model.encode(expected_text, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        return score >= threshold
    except Exception as e:
        logger.warning(f"Semantic scoring fallback to keyword matching due to transformer load failure: {e}")
        return keyword_score(answer, expected_keywords)
