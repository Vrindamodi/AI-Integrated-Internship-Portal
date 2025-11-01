#!/usr/bin/env python3
"""
ai_engine_step1.py

"""
import argparse
import csv
import math
import os
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try optional dependency
try:
    from sentence_transformers import SentenceTransformer, util

    HAS_ST = True
except Exception:
    HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Config / weights
# -----------------------
DEFAULT_WEIGHTS = {
    "semantic": 0.55,
    "qualification": 0.15,
    "cgpa": 0.15,
    "location": 0.08,
    "past_penalty": 0.05,
    "aff_boost": 0.02,
}

QUAL_ORDER = {"10th": 0, "12th": 1, "diploma": 1, "bsc": 2, "bca": 2, "btech": 3, "msc": 4, "mtech": 5, "phd": 6}
CGPA_MIN, CGPA_MAX = 4.0, 10.0
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------
# Data helpers & classes
# -----------------------
def safe_lower(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return str(s).strip().lower()


def parse_semicolon_list(s: Optional[str]) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return [x.strip().lower() for x in str(s).split(";") if x.strip()]


class Candidate:
    def __init__(self, row: pd.Series):
        self.id = str(row.get("id") or row.get("student_id") or "")
        self.name = str(row.get("name") or "")
        self.qualification = safe_lower(row.get("qualification", ""))
        self.skills = parse_semicolon_list(row.get("skills", ""))
        prof = row.get("profile_text")
        self.profile_text = str(prof) if (prof is not None and str(prof).strip() != "") else " ".join(self.skills)
        self.district = safe_lower(row.get("district", ""))
        self.category = safe_lower(row.get("category", "gen"))
        # past_participation used as int or boolean
        try:
            self.past_participation = bool(int(row.get("past_participation", 0)))
        except Exception:
            self.past_participation = str(row.get("past_participation", "")).strip().lower() in ("1", "true", "yes")
        # numeric fields
        try:
            self.cgpa = float(row.get("cgpa")) if row.get("cgpa", "") not in (None, "") else None
        except Exception:
            self.cgpa = None
        try:
            self.distance = float(row.get("distance")) if row.get("distance", "") not in (None, "") else None
        except Exception:
            self.distance = None

        # parse age (robust)
        self.age = None
        try:
            age_raw = row.get("age", None)
            if age_raw is not None and str(age_raw).strip() != "":
                self.age = int(float(str(age_raw).strip()))
        except Exception:
            self.age = None

        # optional fields used by fairness boost (if present)
        self.income = None
        try:
            income_raw = row.get("income", None)
            if income_raw is not None and str(income_raw).strip() != "":
                self.income = float(str(income_raw).strip())
        except Exception:
            self.income = None

        self.gender = safe_lower(row.get("gender", ""))  # may be "" if not present
        # pwd field: accept 1/0, true/false, yes/no
        try:
            self.pwd = bool(int(row.get("pwd", 0)))
        except Exception:
            self.pwd = str(row.get("pwd", "")).strip().lower() in ("1", "true", "yes")



class Internship:
    def __init__(self, row: pd.Series):
        self.id = str(row.get("id") or row.get("internship_id") or "")
        self.org = str(row.get("org") or row.get("organization") or "")
        self.role = str(row.get("role") or row.get("title") or "")
        self.req_skills = parse_semicolon_list(row.get("required_skills", row.get("skills", "")))
        desc = row.get("description")
        self.description = str(desc) if (desc is not None and str(desc).strip() != "") else " ".join(self.req_skills)
        self.min_qualification = safe_lower(row.get("min_qualification", ""))
        try:
            self.capacity = int(row.get("capacity", 1))
        except Exception:
            self.capacity = 1
        self.district = safe_lower(row.get("district", ""))
        self.sector = safe_lower(row.get("sector", ""))
        # reserved_percent for beneficiaries (0-100)
        try:
            rp = int(row.get("reserved_percent", 0))
            self.reserved_percent = max(0, min(100, rp))
        except Exception:
            self.reserved_percent = 0

    def __repr__(self):
        return f"Internship({self.id},{self.role},cap={self.capacity},reserved={self.reserved_percent})"


# -----------------------
# Embedding wrapper (optional)
# -----------------------
class EmbedModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_ST:
            raise RuntimeError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    @staticmethod
    def cos_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # normalize and dot
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.clip(a_norm.dot(b_norm.T), -1.0, 1.0)


# -----------------------
# Scoring components
# -----------------------
def qual_score(cq: str, rq: str) -> float:
    c_ord = QUAL_ORDER.get(cq.lower(), 0)
    r_ord = QUAL_ORDER.get(rq.lower(), 0)
    if c_ord >= r_ord:
        return 1.0
    if r_ord == 0:
        return 0.0
    return max(0.0, 1.0 - (r_ord - c_ord) * 0.3)


def normalize_cgpa(v: Optional[float]) -> float:
    if v is None:
        return 0.0
    v = max(CGPA_MIN, min(CGPA_MAX, v))
    return (v - CGPA_MIN) / (CGPA_MAX - CGPA_MIN)


def location_score(cd: str, idist: str) -> float:
    if not cd or not idist:
        return 0.5
    return 1.0 if cd.strip().lower() == idist.strip().lower() else 0.5


def affirmative_boost(candidate: Candidate, aspirational: Optional[set]) -> float:
    boost = 0.0
    if candidate.category in ("sc", "st"):
        boost += 0.12
    if aspirational and candidate.district in aspirational:
        boost += 0.08
    if candidate.district and "rural" in candidate.district:
        boost += 0.03
    return boost

# -----------------------
# Helper functions: percentiles, fairness boost, eligibility export
# -----------------------
def compute_cgpa_percentiles(candidates: List[Candidate]) -> Dict[str, float]:
    """
    Candidate ID -> CGPA percentile normalized to [0,1].
    If CGPA missing, returns 0.0 for that candidate.
    """
    vals = []
    id_to_val = {}
    for c in candidates:
        try:
            if c.cgpa is not None:
                v = float(c.cgpa)
                vals.append(v)
                id_to_val[c.id] = v
            else:
                id_to_val[c.id] = None
        except Exception:
            id_to_val[c.id] = None
    if not vals:
        return {c.id: 0.0 for c in candidates}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    percentiles = {}
    for cid, v in id_to_val.items():
        if v is None:
            percentiles[cid] = 0.0
        else:
            # count <= v (simple linear scan is fine for small N)
            count_le = 0
            for sv in vals_sorted:
                if sv <= v:
                    count_le += 1
                else:
                    break
            percentiles[cid] = float(count_le) / max(1, n)
    return percentiles


def compute_fairness_boost(
    candidate: Candidate,
    weights: Dict[str, float],
    income_thresholds: Tuple[float, float] = (200000.0, 500000.0),
    aspirational: Optional[set] = None,
) -> float:
    """
    Compute capped additive boost (fractional). Reads cap from weights.get('aff_boost', fallback).
    Returns a float like 0.05 = 5% boost. This is *cap-limited*.
    """
    cap = float(weights.get("aff_boost", 0.02))  # preserve your DEFAULT_WEIGHTS key name
    total = 0.0
    cat = (getattr(candidate, "category", "") or "").strip().lower()
    if cat in ("sc", "st"):
        total += 0.05
    elif cat == "obc":
        total += 0.03
    elif cat == "ews":
        total += 0.03

    # aspirational and rural boosts
    if aspirational and candidate.district and candidate.district in aspirational:
        total += 0.03
    if candidate.district and "rural" in candidate.district.lower():
        total += 0.02

    # income-based boost (only if field exists)
    inc = getattr(candidate, "income", None)
    try:
        if inc is not None and str(inc).strip() != "":
            income_val = float(inc)
            if income_val <= income_thresholds[0]:
                total += 0.05
            elif income_val <= income_thresholds[1]:
                total += 0.03
    except Exception:
        pass

    # small optional boosts
    if getattr(candidate, "pwd", False):
        total += 0.03
    if getattr(candidate, "gender", "").strip().lower() == "female":
        total += 0.02

    return min(total, cap)


def is_eligible(candidate: Candidate, age_min: int = None, age_max: int = None) -> bool:
    """
    Basic eligibility filter.
    - if candidate has 'qualification' that maps to masters/phd (i.e. QUAL_ORDER lookup) we keep them (or treat per policy).
    - This function is conservative: if we cannot parse age, we do not disqualify.
    - You can tune this in main(); minimal and safe for hackathon.
    """
    # qualification rule: if normalized qualification string contains 'phd' or 'master' treat as ineligible for UG-only internships
    q = (getattr(candidate, "qualification", "") or "").strip().lower()
    if q in ("phd", "mtech", "msc", "master", "mphil", "mba", "ms"):
        return False

    # age: check only if attribute exists
    if age_min is not None and age_max is not None:
        age = getattr(candidate, "age", None)
        try:
            if age is not None and str(age).strip() != "":
                a = int(float(age))
                if a < age_min or a > age_max:
                    return False
        except Exception:
            # if never parseable, treat as eligible (safer)
            pass
    return True


def export_ineligible_candidates(ineligible_list: List[Candidate], out: str = "ineligible_candidates.csv"):
    if not ineligible_list:
        return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "name", "qualification", "district", "category"])
        for c in ineligible_list:
            w.writerow([c.id, c.name, getattr(c, "qualification", ""), getattr(c, "district", ""), getattr(c, "category", "")])
    print(f"[+] Exported {len(ineligible_list)} ineligible candidates to {out}")


# -----------------------
# Tie-breaker
# -----------------------
def break_tie(candidates: List[Candidate]) -> Candidate:
    pool = list(candidates)
    no_past = [c for c in pool if not c.past_participation]
    if no_past:
        pool = no_past
    rural = [c for c in pool if c.district and "rural" in c.district]
    if rural:
        pool = rural
    cgs = [c.cgpa for c in pool if c.cgpa is not None]
    if cgs:
        maxcg = max(cgs)
        pool = [c for c in pool if c.cgpa == maxcg]
    dists = [c.distance for c in pool if c.distance is not None]
    if dists:
        mind = min(dists)
        pool = [c for c in pool if c.distance == mind]
    if len(pool) == 1:
        return pool[0]
    # weighted random
    weights = []
    for c in pool:
        w = 1.0
        if c.cgpa is not None:
            w += normalize_cgpa(c.cgpa)
        if not c.past_participation:
            w += 0.3
        if c.district and "rural" in c.district:
            w += 0.2
        weights.append(w)
    total = sum(weights)
    if total <= 0:
        return random.choice(pool)
    probs = [w / total for w in weights]
    chosen = np.random.choice(pool, p=probs)
    return chosen


# -----------------------
# Matching algorithm: hospital/residents (capacity-aware)
# -----------------------
def hospital_residents_matching(
    candidates: List[Candidate],
    internships: List[Internship],
    score_matrix: Dict[str, Dict[str, float]],
    components: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, str]:
    # build candidate preference lists
    prefs = {}
    for c in candidates:
        cid = c.id
        pairs = list(score_matrix.get(cid, {}).items())
        # ranking key: total then semantic then qualification then cgpa
        def rk(item):
            iid, tot = item
            comps = components.get(cid, {}).get(iid, {})
            return (tot, comps.get("semantic", 0), comps.get("qualification", 0), comps.get("cgpa", 0))
        ranked = sorted(pairs, key=rk, reverse=True)
        prefs[cid] = deque([iid for iid, sc in ranked if sc > 0])


    current = {it.id: [] for it in internships}
    capacities = {it.id: it.capacity for it in internships}

    def sc(cid, iid):
        return score_matrix.get(cid, {}).get(iid, 0.0)

    free = deque([c.id for c in candidates])

    while free:
        cid = free.popleft()
        if not prefs[cid]:
            continue
        iid = prefs[cid].popleft()
        cur = current[iid]
        if len(cur) < capacities[iid]:
            cur.append((cid, sc(cid, iid)))
        else:
            worst_idx = min(range(len(cur)), key=lambda k: cur[k][1])
            worst_cid, worst_score = cur[worst_idx]
            proposing_score = sc(cid, iid)
            if proposing_score > worst_score:
                cur[worst_idx] = (cid, proposing_score)
                if prefs[worst_cid]:
                    free.append(worst_cid)
            else:
                if prefs[cid]:
                    free.append(cid)
    matching = {}
    for iid, lst in current.items():
        for cid, s in lst:
            matching[cid] = iid
    return matching


# -----------------------
# Two-phase reserved seats (beneficiaries)
# -----------------------

def two_phase_matching(
    candidates: List[Candidate],
    internships: List[Internship],
    scores: Dict[str, Dict[str, float]],
    components: Dict[str, Dict[str, Dict[str, float]]],
    beneficiary_pred,
) -> Dict[str, str]:
    matched = {}
    int_map = {it.id: it for it in internships}

    # compute reserved seats
    reserved = {it.id: math.floor(it.capacity * it.reserved_percent / 100) for it in internships}
    beneficiaries = [c for c in candidates if beneficiary_pred(c)]

    # Phase 1: beneficiaries only on reserved seats
    temp_interns = []
    for it in internships:
        cap = reserved[it.id]
        if cap > 0:
            tmp = Internship(pd.Series({
                "id": it.id,
                "org": it.org,
                "role": it.role,
                "required_skills": ";".join(it.req_skills),
                "description": it.description,
                "min_qualification": it.min_qualification,
                "district": it.district,
                "sector": it.sector,
                "capacity": cap,
                "reserved_percent": it.reserved_percent
            }))
            temp_interns.append(tmp)

        if temp_interns and beneficiaries:
        # IDs of internships that actually have reserved seats
            temp_ids = {intern.id for intern in temp_interns}

        # Build ben_scores/ben_components only for internships in temp_ids
            ben_scores = {}
            ben_components = {}
            for c in beneficiaries:
                cid = c.id
                cand_scores = scores.get(cid, {})
                cand_comps = components.get(cid, {})
                # Keep only iids that are in the temp_ids set
                filtered_scores = {iid: cand_scores[iid] for iid in cand_scores if iid in temp_ids}
                filtered_comps = {iid: cand_comps[iid] for iid in cand_comps if iid in temp_ids}
                if filtered_scores:
                    ben_scores[cid] = filtered_scores
                    ben_components[cid] = filtered_comps

            # Only run phase1 if any candidate->intern pairs exist for reserved internships
            if ben_scores:
                phase1 = hospital_residents_matching(beneficiaries, temp_interns, ben_scores, ben_components)
                for cid, iid in phase1.items():
                    matched[cid] = iid
                    # decrement the capacity in the original internship map
                    if iid in int_map:
                        int_map[iid].capacity -= 1



    # Phase 2: remaining seats for all
    remaining_candidates = [c for c in candidates if c.id not in matched]
    remaining_interns = [it for it in internships if int_map[it.id].capacity > 0]
    if remaining_candidates and remaining_interns:
        rem_scores = {c.id: {iid: scores[c.id][iid] for iid in scores[c.id] if iid in int_map and int_map[iid].capacity > 0} for c in remaining_candidates}
        rem_components = {c.id: {iid: components[c.id][iid] for iid in components[c.id] if iid in int_map and int_map[iid].capacity > 0} for c in remaining_candidates}
        phase2 = hospital_residents_matching(remaining_candidates, remaining_interns, rem_scores, rem_components)
        for cid, iid in phase2.items():
            matched[cid] = iid
    return matched

# -----------------------
# Replacement: compute_scores_components (merit base + capped hybrid boost -> final)
# -----------------------
def compute_scores_components(
    candidates: List[Candidate],
    internships: List[Internship],
    model: Optional[EmbedModel],
    weights: Dict[str, float],
    aspirational: Optional[set],
):
    """
    Returns:
      scores_base: {cid: {iid: base_score}}  # base merit-only score
      components: {cid: {iid: {component_name: value, 'base': base_score, 'final_boost_applied': applied_boost, 'final': final_score}}}
    Final score used for matching = base + 0.3 * applied_boost  (70% merit dominance).
    """
    use_emb = model is not None
    cand_texts = [c.profile_text for c in candidates]
    int_texts = [it.description for it in internships]

    if use_emb:
        cand_emb = model.encode(cand_texts)
        int_emb = model.encode(int_texts)
        sim_mat = model.cos_sim_matrix(cand_emb, int_emb)
        sim_mat = (sim_mat + 1.0) / 2.0
    else:
        # fallback: skill Jaccard
        sim_mat = np.zeros((len(candidates), len(internships)))
        for i, c in enumerate(candidates):
            for j, it in enumerate(internships):
                sa, sb = set(c.skills), set(it.req_skills)
                if not sa and not sb:
                    sim = 0.0
                else:
                    sim = len(sa & sb) / len(sa | sb)
                sim_mat[i, j] = sim

    # precompute cgpa percentiles
    cgpa_percentiles = compute_cgpa_percentiles(candidates)

    scores_base = {c.id: {} for c in candidates}
    components = {c.id: {} for c in candidates}
    # cgpa_map uses percentile primarily
    cgpa_map = {c.id: cgpa_percentiles.get(c.id, 0.0) for c in candidates}

    for i, c in enumerate(candidates):
        for j, it in enumerate(internships):
            comps = {}

            s_sem = float(sim_mat[i, j])
            comps["semantic"] = s_sem

            s_qual = qual_score(c.qualification, it.min_qualification)
            comps["qualification"] = s_qual

            s_cgpa = cgpa_map.get(c.id, 0.0)
            comps["cgpa_percentile"] = s_cgpa
            comps["cgpa"] = s_cgpa  

            s_loc = location_score(c.district, it.district)
            comps["location"] = s_loc

            s_past = -0.2 if c.past_participation else 0.0
            comps["past_penalty"] = s_past

            # base (merit-only) score
            base = (
                weights.get("semantic", 0.5) * comps["semantic"]
                + weights.get("qualification", 0.12) * comps["qualification"]
                + weights.get("cgpa", 0.08) * comps["cgpa_percentile"]
                + weights.get("location", 0.07) * comps["location"]
                + weights.get("past_penalty", 0.05) * comps["past_penalty"]
            )
            base = float(max(0.0, min(1.0, base)))
            comps["base"] = float(round(base, 6))

            # compute fairness boost (cap respected inside)
            applied_boost = compute_fairness_boost(c, weights, aspirational=aspirational)
            comps["final_boost_applied"] = float(round(applied_boost, 6))
            comps["aff_boost"] = float(round(applied_boost, 6)) 

            # final score (70% merit dominance): final = base + 0.3 * applied_boost
            final = base + 0.3 * applied_boost
            final = float(max(0.0, min(final, 1.2)))
            comps["final"] = float(round(final, 6))

            scores_base[c.id][it.id] = round(float(base), 6)
            components[c.id][it.id] = {k: float(round(v, 6)) for k, v in comps.items()}

    return scores_base, components


# -----------------------
# CSV load / demo generation / export
# -----------------------
def load_students(path: str) -> List[Candidate]:
    df = pd.read_csv(path, dtype=str).fillna("")
    df2 = pd.read_csv(path)
    rows = []
    for i in range(len(df)):
        merged = df.iloc[i].to_dict()
        # pick numeric types from df2 if possible
        try:
            merged["cgpa"] = df2.iloc[i].get("cgpa", merged.get("cgpa", ""))
        except Exception:
            pass
        try:
            merged["distance"] = df2.iloc[i].get("distance", merged.get("distance", ""))
        except Exception:
            pass
        try:
            merged["past_participation"] = df2.iloc[i].get("past_participation", merged.get("past_participation", 0))
        except Exception:
            pass
        rows.append(merged)
    dfm = pd.DataFrame(rows)
    return [Candidate(dfm.iloc[i]) for i in range(len(dfm))]


def load_internships(path: str) -> List[Internship]:
    df = pd.read_csv(path, dtype=str).fillna("")
    df2 = pd.read_csv(path)
    rows = []
    for i in range(len(df)):
        merged = df.iloc[i].to_dict()
        try:
            merged["capacity"] = df2.iloc[i].get("capacity", merged.get("capacity", 1))
        except Exception:
            pass
        try:
            merged["reserved_percent"] = df2.iloc[i].get("reserved_percent", merged.get("reserved_percent", 0))
        except Exception:
            pass
        rows.append(merged)
    dfm = pd.DataFrame(rows)
    return [Internship(dfm.iloc[i]) for i in range(len(dfm))]


def export_matching(
    matching: Dict[str, str],
    candidates: List[Candidate],
    internships: List[Internship],
    components: Dict[str, Dict[str, Dict[str, float]]],
    scores: Dict[str, Dict[str, float]],
    waitlist: Dict[str, List[str]] = None,
    out: str = "placements_step1.csv",
):
    cand_map = {c.id: c for c in candidates}
    int_map = {it.id: it for it in internships}
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["candidate_id", "candidate_name", "internship_id", "org", "role", "total_score",
                  "semantic", "qualification", "cgpa", "location", "past_penalty", "aff_boost","status"]
        w.writerow(header)
        for cid, iid in matching.items():
            c = cand_map.get(cid)
            it = int_map.get(iid)
            comps = components.get(cid, {}).get(iid, {})
            row = [
                cid,
                c.name if c else "",
                iid,
                it.org if it else "",
                it.role if it else "",
                scores.get(cid, {}).get(iid, 0.0),
                comps.get("semantic", 0.0),
                comps.get("qualification", 0.0),
                comps.get("cgpa", 0.0),
                comps.get("location", 0.0),
                comps.get("past_penalty", 0.0),
                comps.get("aff_boost", 0.0),
                "assigned"
            ]
            w.writerow(row)
        if waitlist:
            for iid, cid_list in waitlist.items():
                it = int_map.get(iid)
                for cid in cid_list:
                    c = cand_map.get(cid)
                    comps = components.get(cid, {}).get(iid, {})
                    row = [
                        cid,
                        c.name if c else "",
                        iid,
                        it.org if it else "",
                        it.role if it else "",
                        scores.get(cid, {}).get(iid, 0.0),
                        comps.get("semantic", 0.0),
                        comps.get("qualification", 0.0),
                        comps.get("cgpa", 0.0),
                        comps.get("location", 0.0),
                        comps.get("past_penalty", 0.0),
                        comps.get("aff_boost", 0.0),
                        "waitlist"
                    ]
                    w.writerow(row)
    print(f"[+] Exported placements to {out}")

def build_waitlist_from_scores(
    scores: Dict[str, Dict[str, float]],
    internships: List[Internship],
    candidates: List[Candidate],
    components: Dict[str, Dict[str, Dict[str, float]]],
    matching: Optional[Dict[str,str]] = None,
):
    from collections import defaultdict
    intern_map = {it.id: it for it in internships}
    per_intern = defaultdict(list)

    # reverse mapping internship -> list of (cid, score)
    for cid, targets in scores.items():
        for iid, sc in targets.items():
            per_intern[iid].append((cid, sc))

    assigned_cids = set(matching.keys()) if matching else set()
    waitlist = {}
    for iid, cand_list in per_intern.items():
        it = intern_map.get(iid)
        if not it:
            continue
        # sort by score descending
        sorted_list = sorted(cand_list, key=lambda x: x[1], reverse=True)
        # exclude already assigned candidates
        filtered = [(cid, s) for cid, s in sorted_list if cid not in assigned_cids]
        cap = it.capacity + math.floor(it.capacity * it.reserved_percent / 100)
        # candidates beyond capacity become waitlist
        if len(filtered) > cap:
            waitlist[iid] = [cid for cid, _ in filtered[cap:]]
        else:
            waitlist[iid] = []
    return waitlist


def promote_next_from_waitlist(placements_csv_path: str, internship_id: str, opted_out_cid: Optional[str] = None):
    import pandas as pd

    df = pd.read_csv(placements_csv_path, dtype=str).fillna("")
    # normalize column names just in case
    if "status" not in df.columns:
        raise ValueError("placements CSV must include a 'status' column (use export_matching with waitlist).")

    # If caller provided which candidate opted out, mark their row as opted_out
    if opted_out_cid:
        mask_opt = (df["candidate_id"] == opted_out_cid) & (df["internship_id"] == internship_id) & (df["status"] == "assigned")
        if mask_opt.any():
            df.loc[mask_opt, "status"] = "opted_out"
            print(f"[+] Marked {opted_out_cid} as opted_out for {internship_id}")
        else:
            print(f"[!] Opted-out candidate {opted_out_cid} not found as 'assigned' for {internship_id} in CSV")

    # Find waitlist rows for the internship in CSV order
    wait_idx = df[(df["internship_id"] == internship_id) & (df["status"] == "waitlist")].index.tolist()
    if not wait_idx:
        print(f"[!] No waitlist candidates for {internship_id}")
        return None

    promoted_cid = None
    for idx in wait_idx:
        cand_id = df.at[idx, "candidate_id"]
        # Ensure candidate is not assigned elsewhere already (in case CSV includes other assigned rows)
        assigned_elsewhere = df[(df["candidate_id"] == cand_id) & (df["status"] == "assigned")]
        if not assigned_elsewhere.empty:
            # skip this candidate and continue
            print(f"[i] Skipping {cand_id} because they are already assigned elsewhere")
            continue
        # Promote this candidate
        df.at[idx, "status"] = "assigned"
        promoted_cid = cand_id
        print(f"[+] Promoted {promoted_cid} from waitlist to assigned for {internship_id}")
        break

    # If we did not find anyone suitable
    if promoted_cid is None:
        print(f"[!] No promotable waitlist candidate found for {internship_id}")
        return None

    # Persist changes
    df.to_csv(placements_csv_path, index=False)
    return promoted_cid



# -----------------------
# CLI / orchestrator
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="AI Engine Step1 - core matching")
    parser.add_argument("--students", type=str, default="students_sample.csv")
    parser.add_argument("--internships", type=str, default="internships_sample.csv")
    parser.add_argument("--output", type=str, default="placements_sample.csv", help="output CSV")
    parser.add_argument("--no-embeddings", action="store_true", help="disable embeddings fallback to skill-overlap")
    parser.add_argument("--aspirational", default="", help="semicolon-separated aspirational districts")
    parser.add_argument("--beneficiaries", default="sc;st", help="semicolon-separated beneficiary categories")
    parser.add_argument("--weights", default="", help="override weights key:val,comma separated e.g. semantic:0.6,qualification:0.1")
    args = parser.parse_args()

    # load data (CSV only, demo removed)
    if not args.students or not args.internships:
        parser.error("Both --students and --internships must be provided")
        return
    if not os.path.exists(args.students) or not os.path.exists(args.internships):
        parser.error("CSV files not found")
        return

    candidates = load_students(args.students)
    internships = load_internships(args.internships)
    print(f"[+] Loaded {len(candidates)} candidates and {len(internships)} internships from CSVs")
    print("[DEBUG] Candidate IDs:", [c.id for c in candidates])
    print("[DEBUG] Internship IDs:", [it.id for it in internships])

        # --- Eligibility filter (age/degree) ---
    # Tune age_min and age_max if you want; None means no age enforcement
    AGE_MIN = 21
    AGE_MAX = 30

    eligible_candidates = []
    ineligible_candidates = []
    for c in candidates:
        try:
            if is_eligible(c, age_min=AGE_MIN, age_max=AGE_MAX):
                eligible_candidates.append(c)
            else:
                ineligible_candidates.append(c)
        except Exception:
            ineligible_candidates.append(c)

    print(f"[+] Eligible candidates: {len(eligible_candidates)}")
    if ineligible_candidates:
        print(f"[!] Ineligible candidates (filtered out): {len(ineligible_candidates)}")
        export_ineligible_candidates(ineligible_candidates, out="ineligible_candidates.csv")

    # proceed with only eligible candidates for scoring/matching
    candidates = eligible_candidates


    # weights
    weights = DEFAULT_WEIGHTS.copy()
    if args.weights:
        for kv in args.weights.split(","):
            if ":" in kv:
                k, v = kv.split(":")
                try:
                    weights[k.strip()] = float(v)
                except Exception:
                    pass
    print("[+] Using weights:", weights)

    # aspirational & beneficiaries
    aspir = set([x.strip().lower() for x in args.aspirational.split(";") if x.strip()]) if args.aspirational else set()
    beneficiaries = set([x.strip().lower() for x in args.beneficiaries.split(";") if x.strip()])

    # embeddings
    model = None
    if not args.no_embeddings and HAS_ST:
        print("[*] Loading embedding model (sentence-transformers)...")
        try:
            model = EmbedModel()
            print("[+] Embedding model loaded.")
        except Exception as e:
            print("[!] Failed to load embeddings:", e)
            model = None
    else:
        if not HAS_ST and not args.no_embeddings:
            print("[!] sentence-transformers not installed -> using skill-overlap fallback")
        else:
            print("[*] Embeddings disabled by flag -> using skill-overlap fallback")

    # compute scores
        # compute base scores and components (components contains 'final' per pair)
    scores_base, components = compute_scores_components(candidates, internships, model, weights, aspir)
    print("[+] Scores & components computed (base + final boost stored in components)")

    # Build final scores dict (final = components[cid][iid]['final']) for matching
    scores_final = {c.id: {} for c in candidates}
    for cid in scores_base:
        for iid in scores_base[cid]:
            comps = components.get(cid, {}).get(iid, {})
            final_val = comps.get("final", scores_base[cid].get(iid, 0.0))
            scores_final[cid][iid] = round(float(final_val), 6)

    # beneficiary predicate (unchanged semantics)
    beneficiary_pred = lambda c: (c.category in beneficiaries) or (aspir and c.district in aspir) or (c.district and "rural" in c.district)

    # matching (use scores_final so matching receives boosted final scores)
    matching = two_phase_matching(candidates, internships, scores_final, components, beneficiary_pred)


    #creating waitlist
    auto_waitlist = build_waitlist_from_scores(scores_final, internships, candidates, components, matching=matching)
    export_matching(matching, candidates, internships, components, scores_final, waitlist=auto_waitlist, out=args.output)


    # print brief summary
    per_intern = defaultdict(list)
    for cid, iid in matching.items():
        per_intern[iid].append(cid)
    print("\n--- Summary (sample) ---")
    for it in internships[:10]:
        print(f"{it.id} | {it.role} | cap={it.capacity + math.floor(it.capacity * it.reserved_percent / 100)} | placed={len(per_intern[it.id])}")

    print("\n[+] Done.")


if __name__ == "__main__":
    main()


