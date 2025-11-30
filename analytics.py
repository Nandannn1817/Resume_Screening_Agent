from typing import Dict, List


def compute_resume_quality(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    words = t.split()
    wc = len(words)
    score = 0.0
    # basic heuristics
    if wc < 150:
        score = 30
    elif wc < 400:
        score = 60
    elif wc < 800:
        score = 80
    else:
        score = 70

    sections = ['education', 'experience', 'skills', 'projects', 'certification']
    found = sum(1 for s in sections if s in t)
    score += found * 5
    return min(100, score)


def count_skill_occurrences(jd_text: str, resume_text: str, skills: List[str]) -> tuple[Dict[str, int], Dict[str, int]]:
    jd = jd_text.lower() if jd_text else ''
    res = resume_text.lower() if resume_text else ''
    jd_counts = {}
    res_counts = {}
    for s in skills:
        key = s.lower()
        jd_counts[s] = jd.count(key)
        res_counts[s] = res.count(key)
    return jd_counts, res_counts


def compute_overall_match(candidates: List[dict]) -> float:
    if not candidates:
        return 0.0
    # take top candidate score as overall match for dashboard visualization
    top = max(candidates, key=lambda x: x.get('Score', 0))
    return float(top.get('Score', 0))
