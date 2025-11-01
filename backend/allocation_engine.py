import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

# Try optional dependency
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import Candidate, Internship, Placement, UserProfile, Application
from database import get_db_context

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

class CandidateData:
    """Wrapper class to convert database UserProfile to algorithm format"""
    def __init__(self, profile: UserProfile, user_id: int):
        self.id = str(user_id)  # Use user_id as the candidate ID
        self.user_id = user_id
        self.name = profile.name
        self.qualification = safe_lower(profile.qualification or "")
        self.skills = parse_semicolon_list(profile.skills)
        self.profile_text = profile.profile_text or " ".join(self.skills)
        self.district = safe_lower(profile.district or "")
        self.category = safe_lower(profile.category or "gen")
        self.past_participation = profile.past_participation
        self.cgpa = profile.cgpa
        self.distance = None  # Not available in UserProfile
        self.age = profile.age
        self.income = profile.income
        self.gender = safe_lower(profile.gender or "")
        self.pwd = profile.pwd

class InternshipData:
    """Wrapper class to convert database Internship to algorithm format"""
    def __init__(self, internship: Internship):
        self.id = internship.id
        self.org = internship.org
        self.role = internship.role
        self.req_skills = parse_semicolon_list(internship.required_skills)
        self.description = internship.description or " ".join(self.req_skills)
        self.min_qualification = safe_lower(internship.min_qualification or "")
        self.capacity = internship.capacity
        self.district = safe_lower(internship.district or "")
        self.sector = safe_lower(internship.sector or "")
        self.reserved_percent = internship.reserved_percent

def create_candidates_from_applications(
    applications: List[Application], 
    profiles: List[UserProfile]
) -> List[CandidateData]:
    """Create CandidateData objects from applications and user profiles"""
    profile_map = {p.user_id: p for p in profiles}
    candidates = []
    
    for app in applications:
        if app.user_id in profile_map:
            profile = profile_map[app.user_id]
            candidate = CandidateData(profile, app.user_id)
            candidates.append(candidate)
    
    return candidates

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

def compute_cgpa_percentiles(candidates: List[CandidateData]) -> Dict[str, float]:
    """Candidate ID -> CGPA percentile normalized to [0,1]."""
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
            count_le = 0
            for sv in vals_sorted:
                if sv <= v:
                    count_le += 1
                else:
                    break
            percentiles[cid] = float(count_le) / max(1, n)
    return percentiles

def compute_fairness_boost(
    candidate: CandidateData,
    weights: Dict[str, float],
    income_thresholds: Tuple[float, float] = (200000.0, 500000.0),
    aspirational: Optional[set] = None,
) -> float:
    """Compute capped additive boost (fractional)."""
    cap = float(weights.get("aff_boost", 0.02))
    total = 0.0
    cat = candidate.category.strip().lower()
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

    # income-based boost
    if candidate.income is not None:
        try:
            income_val = float(candidate.income)
            if income_val <= income_thresholds[0]:
                total += 0.05
            elif income_val <= income_thresholds[1]:
                total += 0.03
        except Exception:
            pass

    # small optional boosts
    if candidate.pwd:
        total += 0.03
    if candidate.gender.strip().lower() == "female":
        total += 0.02

    return min(total, cap)

def is_eligible(candidate: CandidateData, age_min: int = None, age_max: int = None) -> bool:
    """Basic eligibility filter for user-driven internship system."""
    # For user-driven system, we allow all qualifications
    # Only check age if specified
    
    # age check
    if age_min is not None and age_max is not None:
        if candidate.age is not None:
            try:
                a = int(candidate.age)
                if a < age_min or a > age_max:
                    return False
            except Exception:
                # If age parsing fails, consider eligible (let the algorithm handle it)
                pass
    
    # All candidates are eligible by default in user-driven system
    return True

# -----------------------
# Matching algorithm
# -----------------------
def hospital_residents_matching(
    candidates: List[CandidateData],
    internships: List[InternshipData],
    score_matrix: Dict[str, Dict[str, float]],
    components: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, str]:
    # build candidate preference lists
    prefs = {}
    for c in candidates:
        cid = c.id
        pairs = list(score_matrix.get(cid, {}).items())
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

def two_phase_matching(
    candidates: List[CandidateData],
    internships: List[InternshipData],
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
            tmp = InternshipData(type('InternshipData', (), {
                'id': it.id,
                'org': it.org,
                'role': it.role,
                'req_skills': it.req_skills,
                'description': it.description,
                'min_qualification': it.min_qualification,
                'district': it.district,
                'sector': it.sector,
                'capacity': cap,
                'reserved_percent': it.reserved_percent
            })())
            temp_interns.append(tmp)

    if temp_interns and beneficiaries:
        temp_ids = {intern.id for intern in temp_interns}
        ben_scores = {}
        ben_components = {}
        for c in beneficiaries:
            cid = c.id
            cand_scores = scores.get(cid, {})
            cand_comps = components.get(cid, {})
            filtered_scores = {iid: cand_scores[iid] for iid in cand_scores if iid in temp_ids}
            filtered_comps = {iid: cand_comps[iid] for iid in cand_comps if iid in temp_ids}
            if filtered_scores:
                ben_scores[cid] = filtered_scores
                ben_components[cid] = filtered_comps

        if ben_scores:
            phase1 = hospital_residents_matching(beneficiaries, temp_interns, ben_scores, ben_components)
            for cid, iid in phase1.items():
                matched[cid] = iid
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

def compute_scores_components(
    candidates: List[CandidateData],
    internships: List[InternshipData],
    model: Optional[EmbedModel],
    weights: Dict[str, float],
    aspirational: Optional[set],
):
    """Compute scores and components for matching."""
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

            # compute fairness boost
            applied_boost = compute_fairness_boost(c, weights, aspirational=aspirational)
            comps["final_boost_applied"] = float(round(applied_boost, 6))
            comps["aff_boost"] = float(round(applied_boost, 6))

            # final score (70% merit dominance)
            final = base + 0.3 * applied_boost
            final = float(max(0.0, min(final, 1.2)))
            comps["final"] = float(round(final, 6))

            scores_base[c.id][it.id] = round(float(base), 6)
            components[c.id][it.id] = {k: float(round(v, 6)) for k, v in comps.items()}

    return scores_base, components

def build_waitlist_from_scores(
    scores: Dict[str, Dict[str, float]],
    internships: List[InternshipData],
    candidates: List[CandidateData],
    components: Dict[str, Dict[str, Dict[str, float]]],
    matching: Optional[Dict[str, str]] = None,
):
    """Build waitlist from scores."""
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
