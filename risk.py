from odds import valid_odd


def fractional_kelly(probability, odd, fraction=0.25, max_fraction=0.25):
    if probability is None or not valid_odd(odd):
        return 0.0

    decimal_odd = float(odd)
    edge = (probability * decimal_odd) - 1
    if edge <= 0:
        return 0.0

    net_odds = decimal_odd - 1
    full_kelly = edge / net_odds
    stake = full_kelly * fraction
    return float(max(0.0, min(stake, max_fraction)))


def stake_size(probability, odd, bankroll=100.0, fraction=0.25, max_fraction=0.25):
    stake_fraction = fractional_kelly(
        probability,
        odd,
        fraction=fraction,
        max_fraction=max_fraction,
    )
    return {
        "stake_fraction": stake_fraction,
        "stake_percent": stake_fraction * 100,
        "stake_units": stake_fraction * bankroll,
        "risk_level": risk_level(stake_fraction),
    }


def risk_level(stake_fraction):
    if stake_fraction <= 0:
        return "none"
    if stake_fraction < 0.03:
        return "low"
    if stake_fraction < 0.08:
        return "medium"
    return "high"
