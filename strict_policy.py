"""
Strict policy utilities for phishing detector.

Defaults tuned from sweep:
    legit_hi = 0.75
    legit_lo = 0.55
    rule_penalty = 0.15
"""

from typing import Tuple, Dict

DEFAULT_LEGIT_HI = 0.75
DEFAULT_LEGIT_LO = 0.55
DEFAULT_RULE_PENALTY = 0.15

def strict_hard_rules(row) -> Tuple[int, Dict[str, float]]:
    """Return (hits, details) for fast hard-rule checks on a single feature row.
    Expects 'row' to be a mapping (e.g., pandas Series) with engineered feature names.
    """
    hits = 0
    details = {}

    def g(key, default=None):
        try:
            v = row[key]
            return v if v is not None else default
        except Exception:
            return default

    # 1) Very long URL
    v = g('URLLength')
    if v is not None and v >= 100:
        hits += 1; details['LONG_URL'] = float(v)

    # 2) Excessive subdomains
    v = g('NoOfSubDomain')
    if v is not None and v >= 3:
        hits += 1; details['MANY_SUBDOMAINS'] = float(v)

    # 3) Special char ratio high
    v = g('SpecialCharRatioInURL')
    if v is not None and v >= 0.08:
        hits += 1; details['SPECIAL_CHAR_RATIO'] = float(v)

    # 4) Digit-heavy
    v = g('DigitRatioInURL')
    if v is not None and v >= 0.35:
        hits += 1; details['DIGIT_HEAVY'] = float(v)

    # 5) Many query params (&)
    v = g('NoOfAmpersandInURL')
    if v is not None and v >= 2:
        hits += 1; details['MANY_PARAMS'] = float(v)

    # 6) IP as domain
    v = g('IsDomainIP')
    try:
        if v is not None and int(v) == 1:
            hits += 1; details['IP_DOMAIN'] = 1.0
    except Exception:
        pass

    # Optional keyword flag columns
    for kw_col in ['Bank','Pay','Crypto']:
        v = g(kw_col)
        try:
            if v is not None and int(v) == 1:
                hits += 1; details[f'KW_{kw_col.upper()}'] = 1.0
        except Exception:
            continue

    return hits, details

def strict_mode_decision(p_legit: float,
                         rule_hits: int,
                         legit_hi: float = DEFAULT_LEGIT_HI,
                         legit_lo: float = DEFAULT_LEGIT_LO,
                         rule_penalty: float = DEFAULT_RULE_PENALTY):
    """Return (label, adjusted_probability).
    label in {"LEGIT","REVIEW","PHISH"}.
    """
    adjusted = max(0.0, float(p_legit) - rule_hits * float(rule_penalty))

    if adjusted >= legit_hi:
        return "LEGIT", adjusted
    elif adjusted < legit_lo:
        return "PHISH", adjusted
    else:
        return "REVIEW", adjusted
