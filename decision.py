from odds import OUTCOMES, OVER_UNDER_OUTCOMES, value_table, valid_odd
from risk import stake_size


MIN_BET_EV = 0.22
MIN_BET_PROBABILITY = 0.42
MIN_SMALL_EV = 0.05
MIN_ODD = 1.65
MAX_ODD = 4.50
KELLY_FRACTION = 0.25
MAX_STAKE_FRACTION = 0.03
MIN_MARKET_EDGE = 0.03


def build_decision(prediction, odds, bankroll=100.0):
    normalized_odds = normalize_odds(odds)
    match_result_odds = normalized_odds["match_result"]
    over_under_odds = normalized_odds["over_under_2_5"]
    probabilities = prediction["probabilities"]
    values = value_table(probabilities, match_result_odds)
    candidate = best_value_candidate(values)
    highest_outcome = prediction["highest_probability_outcome"]
    score_outcome = prediction["predicted_score_outcome"]
    over_under_probabilities = {
        "over": prediction["over_under"]["over"],
        "under": prediction["over_under"]["under"],
    }
    over_under_values = value_table(over_under_probabilities, over_under_odds)
    over_under_candidate = best_value_candidate(
        over_under_values, outcomes=OVER_UNDER_OUTCOMES
    )

    reasons = []
    recommendation = "No Bet"
    conflict = False
    confidence_score = 0.0
    risk = {"stake_fraction": 0.0, "stake_percent": 0.0, "stake_units": 0.0, "risk_level": "none"}

    if candidate is None:
        reasons.append("odds unavailable or invalid")
    else:
        outcome = candidate["outcome"]
        probability = candidate["probability"]
        ev = candidate["ev"]
        odd = candidate["odd"]
        market_probability = candidate["market_probability"]
        market_edge = (
            probability - market_probability
            if market_probability is not None
            else None
        )
        confidence_score = max(0.0, probability * ev) if ev is not None else 0.0

        conflict = score_outcome == "draw" and outcome != "draw"
        if conflict:
            recommendation = "NO BET / CONFLICT SIGNAL"
            reasons.append("predicted score is draw but EV target is not draw")
        elif not valid_odd(odd) or odd < MIN_ODD:
            reasons.append(f"odds too low (<{MIN_ODD}), no value")
        elif odd > MAX_ODD:
            reasons.append(f"odds too high (>{MAX_ODD}), model edge too volatile")
        elif probability < MIN_BET_PROBABILITY:
            reasons.append(f"probability below {MIN_BET_PROBABILITY:.0%}, avoid recommendation")
        elif market_edge is None or market_edge < MIN_MARKET_EDGE:
            reasons.append(f"model edge vs market below {MIN_MARKET_EDGE:.0%}")
        elif ev < MIN_SMALL_EV:
            reasons.append("EV below 0.05, no bet")
        elif ev <= MIN_BET_EV:
            reasons.append(f"EV does not clear professional threshold >{MIN_BET_EV:.2f}")
        else:
            risk = stake_size(
                probability,
                odd,
                bankroll=bankroll,
                fraction=KELLY_FRACTION,
                max_fraction=MAX_STAKE_FRACTION,
            )
            recommendation = recommendation_label(confidence_score, ev)
            reasons.append(
                f"EV {ev:.2f}, probability {probability:.1%}, market edge {market_edge:.1%}"
            )

    over_under_decision = build_over_under_decision(
        over_under_candidate, bankroll=bankroll
    )

    return {
        "recommendation": recommendation,
        "selected_outcome": candidate["outcome"] if candidate else None,
        "selected_label": outcome_label(candidate["outcome"]) if candidate else "-",
        "confidence": confidence_score,
        "confidence_label": confidence_label(confidence_score),
        "risk": risk,
        "reason": "; ".join(reasons) if reasons else "no value signal",
        "values": values,
        "over_under": over_under_decision,
        "over_under_values": over_under_values,
        "model_agreement": {
            "predicted_score": prediction["predicted_score"],
            "score_outcome": score_outcome,
            "highest_probability_outcome": highest_outcome,
            "ev_outcome": candidate["outcome"] if candidate else None,
            "agreement": bool(candidate and not conflict),
        },
    }


def normalize_odds(odds):
    if "match_result" in odds and "over_under_2_5" in odds:
        return odds

    return {
        "match_result": {outcome: odds.get(outcome) for outcome in OUTCOMES},
        "over_under_2_5": {outcome: None for outcome in OVER_UNDER_OUTCOMES},
    }


def build_over_under_decision(candidate, bankroll=100.0):
    reasons = []
    recommendation = "No Bet"
    confidence_score = 0.0
    risk = {"stake_fraction": 0.0, "stake_percent": 0.0, "stake_units": 0.0, "risk_level": "none"}

    if candidate is None:
        reasons.append("odds unavailable or invalid")
    else:
        probability = candidate["probability"]
        ev = candidate["ev"]
        odd = candidate["odd"]
        market_probability = candidate["market_probability"]
        market_edge = (
            probability - market_probability
            if market_probability is not None
            else None
        )
        confidence_score = max(0.0, probability * ev) if ev is not None else 0.0

        if not valid_odd(odd) or odd < MIN_ODD:
            reasons.append(f"odds too low (<{MIN_ODD}), no value")
        elif odd > MAX_ODD:
            reasons.append(f"odds too high (>{MAX_ODD}), model edge too volatile")
        elif probability < MIN_BET_PROBABILITY:
            reasons.append(f"probability below {MIN_BET_PROBABILITY:.0%}, avoid recommendation")
        elif market_edge is None or market_edge < MIN_MARKET_EDGE:
            reasons.append(f"model edge vs market below {MIN_MARKET_EDGE:.0%}")
        elif ev < MIN_SMALL_EV:
            reasons.append("EV below 0.05, no bet")
        elif ev <= MIN_BET_EV:
            reasons.append(f"EV does not clear professional threshold >{MIN_BET_EV:.2f}")
        else:
            risk = stake_size(
                probability,
                odd,
                bankroll=bankroll,
                fraction=KELLY_FRACTION,
                max_fraction=MAX_STAKE_FRACTION,
            )
            recommendation = recommendation_label(confidence_score, ev)
            reasons.append(
                f"EV {ev:.2f}, probability {probability:.1%}, market edge {market_edge:.1%}"
            )

    return {
        "recommendation": recommendation,
        "selected_outcome": candidate["outcome"] if candidate else None,
        "selected_label": over_under_label(candidate["outcome"]) if candidate else "-",
        "confidence": confidence_score,
        "confidence_label": confidence_label(confidence_score),
        "risk": risk,
        "reason": "; ".join(reasons) if reasons else "no value signal",
    }


def best_value_candidate(values, outcomes=OUTCOMES):
    candidates = []
    for outcome in outcomes:
        item = values[outcome]
        if item["ev"] is None:
            continue
        candidates.append({"outcome": outcome, **item})

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["ev"])


def recommendation_label(confidence, ev):
    if confidence >= 0.15 and ev > 0.10:
        return "Strong Bet"
    return "Lean"


def confidence_label(confidence):
    if confidence >= 0.15:
        return "HIGH"
    if confidence >= 0.08:
        return "MEDIUM"
    return "LOW"


def outcome_label(outcome):
    labels = {
        "home": "Home Win",
        "draw": "Draw",
        "away": "Away Win",
    }
    return labels.get(outcome, "-")


def over_under_label(outcome):
    labels = {
        "over": "Over 2.5",
        "under": "Under 2.5",
    }
    return labels.get(outcome, "-")
