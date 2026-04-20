import argparse
import math

import numpy as np
import pandas as pd

from decision import build_decision
from model import load_match_data, predict_match
from odds import OUTCOMES, valid_odd


RULE_MIN_PROBABILITY = 0.45
RULE_MIN_ODD = 1.50
RULE_STAKE_FRACTION = 0.01


def run_backtest(match_count=1000, min_history=1000):
    df = load_match_data().reset_index(drop=True)
    if len(df) <= min_history:
        raise ValueError("Dataset terlalu kecil untuk walk-forward backtest")

    start = max(min_history, len(df) - match_count)
    test = df.iloc[start:].copy()

    rule_bets = []
    ev_bets = []
    prediction_rows = []

    for idx, match in test.iterrows():
        train = df.iloc[:idx]
        prediction = predict_match(train, match["HomeTeam"], match["AwayTeam"])
        actual = actual_outcome(match)
        odds = row_odds(match)

        prediction_rows.append(
            {
                "actual": actual,
                "predicted": prediction["highest_probability_outcome"],
                "confidence": max(prediction["probabilities"].values()),
            }
        )

        rule_bet = rule_based_bet(prediction, odds)
        if rule_bet:
            rule_bets.append(settle_bet(rule_bet, actual))

        ev_decision = build_decision(prediction, odds)
        ev_bet = ev_based_bet(ev_decision, odds)
        if ev_bet:
            ev_bets.append(settle_bet(ev_bet, actual))

    return {
        "meta": {
            "dataset_rows": len(df),
            "test_rows": len(test),
            "start_date": str(test.iloc[0].get("Date")),
            "end_date": str(test.iloc[-1].get("Date")),
            "min_history": min_history,
        },
        "prediction": prediction_summary(prediction_rows),
        "rule_based": strategy_summary(rule_bets),
        "ev_based": strategy_summary(ev_bets),
    }


def rule_based_bet(prediction, odds):
    outcome = prediction["highest_probability_outcome"]
    probability = prediction["probabilities"][outcome]
    odd = odds.get(outcome)

    if probability < RULE_MIN_PROBABILITY:
        return None
    if not valid_odd(odd) or odd < RULE_MIN_ODD:
        return None

    return {
        "strategy": "rule_based",
        "outcome": outcome,
        "odd": odd,
        "probability": probability,
        "stake_fraction": RULE_STAKE_FRACTION,
    }


def ev_based_bet(decision, odds):
    if decision["recommendation"] not in ("Lean", "Strong Bet"):
        return None

    outcome = decision["selected_outcome"]
    if outcome not in OUTCOMES:
        return None

    stake_fraction = decision["risk"]["stake_fraction"]
    if stake_fraction <= 0:
        return None

    return {
        "strategy": "ev_based",
        "outcome": outcome,
        "odd": odds.get(outcome),
        "probability": decision["values"][outcome]["probability"],
        "ev": decision["values"][outcome]["ev"],
        "stake_fraction": stake_fraction,
    }


def settle_bet(bet, actual_outcome_value):
    stake = bet["stake_fraction"]
    won = bet["outcome"] == actual_outcome_value
    profit_units = stake * (bet["odd"] - 1) if won else -stake

    settled = dict(bet)
    settled.update(
        {
            "won": won,
            "profit_units": profit_units,
            "roi_per_staked_unit": profit_units / stake if stake > 0 else 0.0,
        }
    )
    return settled


def strategy_summary(bets):
    if not bets:
        return {
            "bets": 0,
            "hit_rate": 0.0,
            "total_staked_units": 0.0,
            "profit_units": 0.0,
            "roi": 0.0,
            "max_drawdown_units": 0.0,
            "sharpe": 0.0,
            "avg_odds": 0.0,
            "avg_stake_percent": 0.0,
        }

    frame = pd.DataFrame(bets)
    equity = frame["profit_units"].cumsum()
    running_peak = equity.cummax().clip(lower=0)
    drawdown = running_peak - equity
    returns = frame["profit_units"].to_numpy(dtype=float)

    total_staked = frame["stake_fraction"].sum()
    profit = frame["profit_units"].sum()
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns, ddof=1) > 0:
        sharpe = np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(len(returns))

    return {
        "bets": int(len(frame)),
        "hit_rate": float(frame["won"].mean()),
        "total_staked_units": float(total_staked),
        "profit_units": float(profit),
        "roi": float(profit / total_staked) if total_staked > 0 else 0.0,
        "max_drawdown_units": float(drawdown.max()),
        "sharpe": float(sharpe),
        "avg_odds": float(frame["odd"].mean()),
        "avg_stake_percent": float(frame["stake_fraction"].mean() * 100),
    }


def prediction_summary(rows):
    frame = pd.DataFrame(rows)
    return {
        "accuracy": float((frame["actual"] == frame["predicted"]).mean()),
        "avg_confidence": float(frame["confidence"].mean()),
        "predicted_counts": frame["predicted"].value_counts().to_dict(),
        "actual_counts": frame["actual"].value_counts().to_dict(),
    }


def row_odds(match):
    columns = {"home": "B365H", "draw": "B365D", "away": "B365A"}
    odds = {}
    for outcome, column in columns.items():
        value = pd.to_numeric(match.get(column), errors="coerce")
        odds[outcome] = float(value) if pd.notna(value) and float(value) > 1 else None
    return odds


def actual_outcome(match):
    if match["FTHG"] > match["FTAG"]:
        return "home"
    if match["FTHG"] < match["FTAG"]:
        return "away"
    return "draw"


def print_report(result):
    meta = result["meta"]
    print("BACKTEST walk-forward_no_future_leakage")
    print(f"DATASET_ROWS {meta['dataset_rows']}")
    print(f"TEST_ROWS {meta['test_rows']}")
    print(f"DATE_RANGE {meta['start_date']} to {meta['end_date']}")
    print(f"MIN_HISTORY {meta['min_history']}")

    pred = result["prediction"]
    print("\nPREDICTION")
    print(f"ACCURACY {pred['accuracy']:.4f}")
    print(f"AVG_CONFIDENCE {pred['avg_confidence']:.4f}")
    print(f"PREDICTED_COUNTS {pred['predicted_counts']}")
    print(f"ACTUAL_COUNTS {pred['actual_counts']}")

    print("\nSTRATEGY_COMPARISON")
    print_strategy("RULE_BASED", result["rule_based"])
    print_strategy("EV_BASED", result["ev_based"])


def print_strategy(name, summary):
    print(f"{name}_BETS {summary['bets']}")
    print(f"{name}_HIT_RATE {summary['hit_rate']:.4f}")
    print(f"{name}_TOTAL_STAKED_UNITS {summary['total_staked_units']:.4f}")
    print(f"{name}_PROFIT_UNITS {summary['profit_units']:.4f}")
    print(f"{name}_ROI {summary['roi']:.4f}")
    print(f"{name}_MAX_DRAWDOWN_UNITS {summary['max_drawdown_units']:.4f}")
    print(f"{name}_SHARPE {summary['sharpe']:.4f}")
    print(f"{name}_AVG_ODDS {summary['avg_odds']:.4f}")
    print(f"{name}_AVG_STAKE_PERCENT {summary['avg_stake_percent']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--min-history", type=int, default=1000)
    args = parser.parse_args()

    result = run_backtest(match_count=args.matches, min_history=args.min_history)
    print_report(result)


if __name__ == "__main__":
    main()
