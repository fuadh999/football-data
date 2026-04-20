import math

import pandas as pd


OUTCOMES = ("home", "draw", "away")
OVER_UNDER_OUTCOMES = ("over", "under")
ODDS_COLUMNS = {
    "home": "B365H",
    "draw": "B365D",
    "away": "B365A",
}
OVER_UNDER_COLUMNS = {
    "over": ("B365>2.5", "B365O2.5", "B365O25", "B365O"),
    "under": ("B365<2.5", "B365U2.5", "B365U25", "B365U"),
}


def get_match_odds(df, home_team, away_team):
    match = df[(df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)]
    if match.empty:
        return {
            "match_result": {outcome: None for outcome in OUTCOMES},
            "over_under_2_5": {outcome: None for outcome in OVER_UNDER_OUTCOMES},
        }

    row = match.iloc[-1]
    return {
        "match_result": extract_market_odds(row, ODDS_COLUMNS, OUTCOMES),
        "over_under_2_5": extract_over_under_odds(row),
    }


def extract_market_odds(row, columns, outcomes):
    return {
        outcome: clean_odd(row[column]) if column in row.index else None
        for outcome, column in columns.items()
        if outcome in outcomes
    }


def extract_over_under_odds(row):
    return {
        outcome: clean_odd(first_available_value(row, columns))
        for outcome, columns in OVER_UNDER_COLUMNS.items()
    }


def first_available_value(row, columns):
    for column in columns:
        if column in row.index:
            return row[column]
    return None


def clean_odd(value):
    odd = pd.to_numeric(value, errors="coerce")
    if odd is None or not math.isfinite(float(odd)) or float(odd) <= 1:
        return None
    return float(odd)


def implied_probability(odd):
    if not valid_odd(odd):
        return None
    return 1 / float(odd)


def market_probabilities(odds):
    implied = {
        outcome: implied_probability(odd)
        for outcome, odd in odds.items()
    }
    valid_values = [value for value in implied.values() if value is not None]
    total = sum(valid_values)
    if total <= 0:
        return {outcome: None for outcome in odds}

    return {
        outcome: (value / total if value is not None else None)
        for outcome, value in implied.items()
    }


def expected_value(probability, odd):
    if probability is None or not valid_odd(odd):
        return None
    return (float(probability) * float(odd)) - 1


def value_table(probabilities, odds):
    market = market_probabilities(odds)
    return {
        outcome: {
            "probability": probabilities.get(outcome),
            "odd": odds.get(outcome),
            "implied_probability": implied_probability(odds.get(outcome)),
            "market_probability": market.get(outcome),
            "ev": expected_value(probabilities.get(outcome), odds.get(outcome)),
        }
        for outcome in odds
    }


def valid_odd(odd):
    return odd is not None and math.isfinite(float(odd)) and float(odd) > 1
