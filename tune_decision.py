import argparse
import itertools
import math

import numpy as np
import pandas as pd

from model import load_match_data, predict_match
from odds import expected_value, valid_odd
from risk import fractional_kelly


OUTCOMES = ("home", "draw", "away")


def build_prediction_cache(match_count=3000, min_history=1000):
    df = load_match_data().reset_index(drop=True)
    start = max(min_history, len(df) - match_count)
    test = df.iloc[start:].copy()
    rows = []

    for idx, match in test.iterrows():
        prediction = predict_match(df.iloc[:idx], match["HomeTeam"], match["AwayTeam"])
        actual = actual_outcome(match)
        odds = row_odds(match)

        row = {
            "idx": idx,
            "date": match.get("Date"),
            "home_team": match["HomeTeam"],
            "away_team": match["AwayTeam"],
            "actual": actual,
            "predicted_score": prediction["predicted_score"],
            "score_outcome": prediction["predicted_score_outcome"],
            "highest_outcome": prediction["highest_probability_outcome"],
        }

        for outcome in OUTCOMES:
            probability = prediction["probabilities"][outcome]
            odd = odds[outcome]
            row[f"p_{outcome}"] = probability
            row[f"odd_{outcome}"] = odd
            row[f"ev_{outcome}"] = expected_value(probability, odd)

        rows.append(row)

    return pd.DataFrame(rows)


def tune(cache):
    grid = {
        "min_ev": [0.10, 0.14, 0.18, 0.22, 0.28],
        "min_probability": [0.30, 0.34, 0.38, 0.42],
        "min_odd": [1.50, 1.65, 1.80],
        "max_odd": [3.50, 4.50, 6.00],
        "kelly_fraction": [0.10, 0.15, 0.20, 0.25],
        "max_stake": [0.03, 0.05, 0.08],
        "require_agreement": [True, False],
        "avoid_score_draw": [True, False],
    }

    results = []
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        bets = simulate_strategy_vectorized(cache, params)
        summary = strategy_summary(bets)
        summary.update(params)
        summary["score"] = rank_score(summary)
        results.append(summary)

    return pd.DataFrame(results)


def simulate_strategy_vectorized(cache, params):
    candidate_frames = []
    for outcome in OUTCOMES:
        frame = pd.DataFrame(
            {
                "outcome": outcome,
                "actual": cache["actual"].to_numpy(),
                "probability": cache[f"p_{outcome}"].to_numpy(dtype=float),
                "odd": cache[f"odd_{outcome}"].to_numpy(dtype=float),
                "ev": cache[f"ev_{outcome}"].to_numpy(dtype=float),
                "highest_outcome": cache["highest_outcome"].to_numpy(),
                "score_outcome": cache["score_outcome"].to_numpy(),
            }
        )
        candidate_frames.append(frame)

    candidates = pd.concat(candidate_frames, ignore_index=True)
    mask = (
        candidates["ev"].notna()
        & candidates["odd"].notna()
        & (candidates["odd"] > 1)
        & (candidates["ev"] >= params["min_ev"])
        & (candidates["probability"] >= params["min_probability"])
        & (candidates["odd"] >= params["min_odd"])
        & (candidates["odd"] <= params["max_odd"])
    )
    if params["require_agreement"]:
        mask &= candidates["outcome"] == candidates["highest_outcome"]
    if params["avoid_score_draw"]:
        mask &= (candidates["score_outcome"] != "draw") | (candidates["outcome"] == "draw")

    candidates = candidates[mask].copy()
    if candidates.empty:
        return []

    candidates["_match_order"] = np.tile(np.arange(len(cache)), len(OUTCOMES))[mask.to_numpy()]
    candidates = candidates.sort_values(["_match_order", "ev"], ascending=[True, False])
    bets = candidates.drop_duplicates("_match_order", keep="first").copy()

    net_odds = bets["odd"] - 1
    edge = (bets["probability"] * bets["odd"]) - 1
    full_kelly = edge / net_odds
    bets["stake"] = (full_kelly * params["kelly_fraction"]).clip(
        lower=0,
        upper=params["max_stake"],
    )
    bets = bets[bets["stake"] > 0].copy()
    if bets.empty:
        return []

    bets["won"] = bets["outcome"] == bets["actual"]
    bets["profit"] = np.where(
        bets["won"],
        bets["stake"] * (bets["odd"] - 1),
        -bets["stake"],
    )
    return bets[
        ["outcome", "actual", "probability", "odd", "ev", "stake", "won", "profit"]
    ].to_dict("records")


def simulate_strategy(cache, params):
    bets = []
    for _, row in cache.iterrows():
        candidate = best_candidate(row, params)
        if candidate is None:
            continue

        outcome = candidate["outcome"]
        probability = candidate["probability"]
        odd = candidate["odd"]
        stake = fractional_kelly(
            probability,
            odd,
            fraction=params["kelly_fraction"],
            max_fraction=params["max_stake"],
        )
        if stake <= 0:
            continue

        won = outcome == row["actual"]
        profit = stake * (odd - 1) if won else -stake
        bets.append(
            {
                "outcome": outcome,
                "actual": row["actual"],
                "probability": probability,
                "odd": odd,
                "ev": candidate["ev"],
                "stake": stake,
                "won": won,
                "profit": profit,
            }
        )
    return bets


def best_candidate(row, params):
    candidates = []
    for outcome in OUTCOMES:
        probability = row[f"p_{outcome}"]
        odd = row[f"odd_{outcome}"]
        ev = row[f"ev_{outcome}"]

        if ev is None or pd.isna(ev):
            continue
        if not valid_odd(odd):
            continue
        if ev < params["min_ev"]:
            continue
        if probability < params["min_probability"]:
            continue
        if odd < params["min_odd"] or odd > params["max_odd"]:
            continue
        if params["require_agreement"] and outcome != row["highest_outcome"]:
            continue
        if params["avoid_score_draw"] and row["score_outcome"] == "draw" and outcome != "draw":
            continue

        candidates.append(
            {
                "outcome": outcome,
                "probability": probability,
                "odd": odd,
                "ev": ev,
            }
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["ev"])


def strategy_summary(bets):
    if not bets:
        return empty_summary()

    frame = pd.DataFrame(bets)
    equity = frame["profit"].cumsum()
    running_peak = equity.cummax().clip(lower=0)
    drawdown = running_peak - equity
    total_staked = frame["stake"].sum()
    profit = frame["profit"].sum()
    returns = frame["profit"].to_numpy(dtype=float)
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns, ddof=1) > 0:
        sharpe = np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(len(returns))

    return {
        "bets": int(len(frame)),
        "hit_rate": float(frame["won"].mean()),
        "profit": float(profit),
        "staked": float(total_staked),
        "roi": float(profit / total_staked) if total_staked > 0 else 0.0,
        "max_drawdown": float(drawdown.max()),
        "sharpe": float(sharpe),
        "avg_odds": float(frame["odd"].mean()),
        "avg_ev": float(frame["ev"].mean()),
        "avg_stake_percent": float(frame["stake"].mean() * 100),
    }


def empty_summary():
    return {
        "bets": 0,
        "hit_rate": 0.0,
        "profit": 0.0,
        "staked": 0.0,
        "roi": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "avg_odds": 0.0,
        "avg_ev": 0.0,
        "avg_stake_percent": 0.0,
    }


def rank_score(summary):
    if summary["bets"] < 25:
        return -999
    return (
        summary["roi"] * 1.00
        + summary["sharpe"] * 0.08
        - summary["max_drawdown"] * 0.18
        + min(summary["bets"], 250) / 250 * 0.04
    )


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


def print_report(cache, results):
    best = results.sort_values("score", ascending=False).head(15)
    stable = results[results["bets"] >= 50].sort_values(
        ["roi", "sharpe"], ascending=False
    ).head(15)

    print("CACHE_ROWS", len(cache))
    print("DATE_RANGE", str(cache.iloc[0]["date"]), "to", str(cache.iloc[-1]["date"]))
    print("ACTUAL_COUNTS", cache["actual"].value_counts().to_dict())
    print("PREDICTED_COUNTS", cache["highest_outcome"].value_counts().to_dict())

    columns = [
        "score",
        "bets",
        "hit_rate",
        "profit",
        "staked",
        "roi",
        "max_drawdown",
        "sharpe",
        "avg_odds",
        "avg_ev",
        "avg_stake_percent",
        "min_ev",
        "min_probability",
        "min_odd",
        "max_odd",
        "kelly_fraction",
        "max_stake",
        "require_agreement",
        "avoid_score_draw",
    ]
    print("\nTOP_15_RANKED")
    print(best[columns].round(4).to_string(index=False))
    print("\nTOP_15_ROI_MIN_50_BETS")
    print(stable[columns].round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=3000)
    parser.add_argument("--min-history", type=int, default=1000)
    args = parser.parse_args()

    cache = build_prediction_cache(
        match_count=args.matches,
        min_history=args.min_history,
    )
    results = tune(cache)
    print_report(cache, results)


if __name__ == "__main__":
    main()
