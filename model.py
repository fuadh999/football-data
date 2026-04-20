from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from team_aliases import canonical_team_name
from update_data import ensure_dataset_current


DATA_PATH = Path("data/data.csv")
DATA_URL = "https://raw.githubusercontent.com/fuadh999/football-data/main/data/data.csv"
REQUIRED_COLUMNS = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
OUTCOMES = ("home", "draw", "away")


MODEL_HOOKS = {
    "elo_rating": None,
    "player_data": None,
    "injury_news": None,
}


def load_match_data(path=DATA_PATH, fallback_url=DATA_URL):
    data_path = Path(path)
    ensure_dataset_current(output_path=data_path, quiet=True)
    if data_path.exists() and data_path.stat().st_size > 0:
        df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(fallback_url)

    return prepare_match_data(df)


def prepare_match_data(df):
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Data tidak lengkap. Kolom hilang: {', '.join(sorted(missing))}")

    prepared = df.copy()
    prepared = prepared.dropna(subset=list(REQUIRED_COLUMNS))
    prepared["FTHG"] = pd.to_numeric(prepared["FTHG"], errors="coerce")
    prepared["FTAG"] = pd.to_numeric(prepared["FTAG"], errors="coerce")
    prepared = prepared.dropna(subset=["FTHG", "FTAG"])
    prepared["HomeTeam"] = prepared["HomeTeam"].apply(canonical_team_name)
    prepared["AwayTeam"] = prepared["AwayTeam"].apply(canonical_team_name)

    if "Date" in prepared.columns:
        prepared["_match_date"] = pd.to_datetime(
            prepared["Date"], errors="coerce", dayfirst=True
        )
        prepared["_season"] = prepared["_match_date"].apply(match_season)
        prepared = prepared.sort_values(["_match_date"], na_position="first")

    return prepared.reset_index(drop=True)


def get_teams(df):
    return sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))


def predict_match(
    df,
    home_team,
    away_team,
    max_goals=5,
    decay=0.975,
    rho=-0.08,
    calibration_temperature=1.15,
    calibration_shrinkage=0.10,
    favorite_cap=0.66,
    draw_close_margin=0.05,
    min_draw_probability=0.34,
):
    context = infer_match_context(df, home_team, away_team)
    profile = league_profile(
        df,
        decay=decay,
        league=context["league"],
        season=context["season"],
    )
    home_strength = team_strength(
        df, home_team, profile, decay=decay, league=context["league"]
    )
    away_strength = team_strength(
        df, away_team, profile, decay=decay, league=context["league"]
    )

    home_xg, away_xg = expected_goals(profile, home_strength, away_strength)
    baseline_matrix = poisson_score_matrix(home_xg, away_xg, max_goals=max_goals)
    score_matrix = dixon_coles_adjustment(
        baseline_matrix, home_xg, away_xg, rho=rho
    )
    raw_probabilities = outcome_probabilities(score_matrix)
    probabilities = calibrate_probabilities(
        raw_probabilities,
        profile["outcome_prior"],
        temperature=calibration_temperature,
        shrinkage=calibration_shrinkage,
        favorite_cap=favorite_cap,
    )
    probabilities = calibrate_draw_probability(
        probabilities,
        profile,
        home_xg,
        away_xg,
    )
    over25, under25 = over_under_probability(score_matrix, line=2.5)

    predicted_score = most_likely_score(score_matrix)
    highest_probability_outcome = select_primary_outcome(
        probabilities,
        predicted_score,
        draw_close_margin=draw_close_margin,
        min_draw_probability=min_draw_probability,
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "probabilities": _probability_dict(probabilities),
        "raw_probabilities": _probability_dict(raw_probabilities),
        "predicted_score": predicted_score,
        "predicted_score_outcome": score_outcome(predicted_score),
        "highest_probability_outcome": highest_probability_outcome,
        "over_under": {
            "line": 2.5,
            "over": over25,
            "under": under25,
            "prediction": "over" if over25 >= under25 else "under",
        },
        "score_matrix": score_matrix,
        "score_matrix_percent": np.round(score_matrix * 100, 2),
        "home_strength": home_strength,
        "away_strength": away_strength,
        "home_advantage": profile["home_advantage"],
        "league": context["league"],
        "season": context["season"],
        "model": {
            "baseline": "Poisson",
            "adjustment": "Dixon-Coles low-score correction",
            "time_weighting_decay": decay,
            "calibration": "league/season prior + draw calibration + favorite cap",
            "hooks": MODEL_HOOKS,
        },
    }


def infer_match_context(df, home_team, away_team, lookback=30):
    matches = df[
        (df["HomeTeam"].isin([home_team, away_team]))
        | (df["AwayTeam"].isin([home_team, away_team]))
    ].tail(lookback)

    league = None
    if "Div" in matches.columns and not matches.empty:
        mode = matches["Div"].dropna().mode()
        if not mode.empty:
            league = mode.iloc[0]

    season = None
    if "_season" in df.columns and not df.empty:
        latest = df["_season"].dropna()
        if not latest.empty:
            season = int(latest.iloc[-1])

    return {"league": league, "season": season}


def league_profile(df, decay=0.965, league=None, season=None, min_matches=120):
    scoped = scope_matches(df, league=league, season=season, min_matches=min_matches)
    weights = recency_weights(len(scoped), decay=decay)
    home_goals = weighted_average(scoped["FTHG"], weights)
    away_goals = weighted_average(scoped["FTAG"], weights)
    avg_goals = (home_goals + away_goals) / 2

    home_wins = weighted_average((scoped["FTHG"] > scoped["FTAG"]).astype(float), weights)
    draws = weighted_average((scoped["FTHG"] == scoped["FTAG"]).astype(float), weights)
    away_wins = weighted_average((scoped["FTHG"] < scoped["FTAG"]).astype(float), weights)

    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "avg_goals": avg_goals,
        "home_advantage": np.sqrt(home_goals / away_goals) if away_goals > 0 else 1.0,
        "outcome_prior": normalize([home_wins, draws, away_wins]),
        "league": league,
        "season": season,
        "matches": int(len(scoped)),
    }


def scope_matches(df, league=None, season=None, min_matches=120):
    scoped = df
    if league is not None and "Div" in scoped.columns:
        league_scoped = scoped[scoped["Div"] == league]
        if len(league_scoped) >= min_matches:
            scoped = league_scoped

    if season is not None and "_season" in scoped.columns:
        season_scoped = scoped[scoped["_season"] >= season - 1]
        if len(season_scoped) >= min_matches:
            scoped = season_scoped

    return scoped


def team_strength(df, team, profile, decay=0.965, league=None, min_matches=10):
    matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    if league is not None and "Div" in matches.columns:
        league_matches = matches[matches["Div"] == league]
        if len(league_matches) >= min_matches:
            matches = league_matches

    if matches.empty:
        return {
            "attack": 1.0,
            "defense": 1.0,
            "matches": 0,
            "weighted_goals_for": profile["avg_goals"],
            "weighted_goals_against": profile["avg_goals"],
        }

    goals_for = []
    goals_against = []
    for _, row in matches.iterrows():
        if row["HomeTeam"] == team:
            goals_for.append(row["FTHG"])
            goals_against.append(row["FTAG"])
        else:
            goals_for.append(row["FTAG"])
            goals_against.append(row["FTHG"])

    weights = recency_weights(len(matches), decay=decay)
    weighted_for = weighted_average(goals_for, weights)
    weighted_against = weighted_average(goals_against, weights)

    return {
        "attack": safe_ratio(weighted_for, profile["avg_goals"]),
        "defense": safe_ratio(weighted_against, profile["avg_goals"]),
        "matches": int(len(matches)),
        "weighted_goals_for": weighted_for,
        "weighted_goals_against": weighted_against,
    }


def expected_goals(profile, home_strength, away_strength):
    home_xg = (
        profile["avg_goals"]
        * home_strength["attack"]
        * away_strength["defense"]
        * profile["home_advantage"]
    )
    away_xg = (
        profile["avg_goals"]
        * away_strength["attack"]
        * home_strength["defense"]
        / profile["home_advantage"]
    )
    return float(np.clip(home_xg, 0.10, 5.00)), float(np.clip(away_xg, 0.10, 5.00))


def poisson_score_matrix(home_xg, away_xg, max_goals=5):
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, home_xg), poisson.pmf(goals, away_xg))
    return normalize_matrix(matrix)


def dixon_coles_adjustment(score_matrix, home_xg, away_xg, rho=-0.08):
    adjusted = score_matrix.copy()
    low_score_factors = {
        (0, 0): 1 - (home_xg * away_xg * rho),
        (0, 1): 1 + (home_xg * rho),
        (1, 0): 1 + (away_xg * rho),
        (1, 1): 1 - rho,
    }

    for (home_goals, away_goals), factor in low_score_factors.items():
        if home_goals < adjusted.shape[0] and away_goals < adjusted.shape[1]:
            adjusted[home_goals, away_goals] *= max(0.01, factor)

    return normalize_matrix(adjusted)


def outcome_probabilities(score_matrix):
    return normalize(
        [
            np.tril(score_matrix, -1).sum(),
            np.diag(score_matrix).sum(),
            np.triu(score_matrix, 1).sum(),
        ]
    )


def over_under_probability(score_matrix, line=2.5):
    over = 0.0
    for home_goals in range(score_matrix.shape[0]):
        for away_goals in range(score_matrix.shape[1]):
            if home_goals + away_goals > line:
                over += score_matrix[home_goals, away_goals]

    return float(over), float(1 - over)


def calibrate_probabilities(
    probabilities,
    priors,
    temperature=1.12,
    shrinkage=0.10,
    favorite_cap=0.70,
):
    scaled = temperature_scale(probabilities, temperature=temperature)
    calibrated = normalize((1 - shrinkage) * scaled + shrinkage * normalize(priors))

    favorite_idx = int(np.argmax(calibrated))
    if calibrated[favorite_idx] > favorite_cap:
        excess = calibrated[favorite_idx] - favorite_cap
        calibrated[favorite_idx] = favorite_cap
        other_idx = [idx for idx in range(len(calibrated)) if idx != favorite_idx]
        other_total = calibrated[other_idx].sum()
        if other_total > 0:
            calibrated[other_idx] += excess * calibrated[other_idx] / other_total

    return normalize(calibrated)


def calibrate_draw_probability(probabilities, profile, home_xg, away_xg):
    calibrated = np.array(probabilities, dtype=float)
    league_draw = float(profile["outcome_prior"][1])
    xg_gap = abs(home_xg - away_xg)
    total_xg = home_xg + away_xg
    closeness = float(np.exp(-1.25 * xg_gap))
    low_total = float(np.exp(-0.45 * max(total_xg - 2.2, 0)))

    draw_anchor = np.clip(
        league_draw + (0.08 * closeness * low_total) - (0.03 * max(total_xg - 2.8, 0)),
        0.18,
        0.36,
    )
    blend = 0.18 + (0.12 * closeness)
    calibrated[1] = (1 - blend) * calibrated[1] + blend * draw_anchor
    remaining = max(1 - calibrated[1], 1e-9)
    side_total = calibrated[0] + calibrated[2]
    if side_total > 0:
        calibrated[0] = remaining * calibrated[0] / side_total
        calibrated[2] = remaining * calibrated[2] / side_total

    return normalize(calibrated)


def temperature_scale(probabilities, temperature=1.12):
    clipped = np.clip(np.array(probabilities, dtype=float), 1e-9, 1)
    logits = np.log(clipped)
    scaled = np.exp(logits / max(temperature, 1e-6))
    return normalize(scaled)


def most_likely_score(score_matrix):
    home_goals, away_goals = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    return f"{home_goals}-{away_goals}"


def select_primary_outcome(
    probabilities,
    predicted_score,
    draw_close_margin=0.06,
    min_draw_probability=0.27,
):
    primary = OUTCOMES[int(np.argmax(probabilities))]
    if score_outcome(predicted_score) != "draw":
        return primary

    draw_probability = float(probabilities[1])
    favorite_probability = float(np.max(probabilities))
    if (
        primary != "draw"
        and draw_probability >= min_draw_probability
        and favorite_probability - draw_probability <= draw_close_margin
    ):
        return "draw"

    return primary


def score_outcome(score):
    home_goals, away_goals = [int(value) for value in score.split("-")]
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def recency_weights(length, decay=0.965):
    if length <= 0:
        return np.array([], dtype=float)
    ages = np.arange(length - 1, -1, -1, dtype=float)
    return np.power(decay, ages)


def match_season(date_value):
    if pd.isna(date_value):
        return None
    return int(date_value.year + 1 if date_value.month >= 7 else date_value.year)


def weighted_average(values, weights):
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)
    if len(values) == 0 or weights.sum() <= 0:
        return 0.0
    return float(np.average(values, weights=weights))


def safe_ratio(value, baseline):
    if baseline is None or baseline <= 0 or pd.isna(baseline):
        return 1.0
    return float(value / baseline)


def normalize(values):
    arr = np.array(values, dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def normalize_matrix(matrix):
    total = matrix.sum()
    if total <= 0:
        return np.ones_like(matrix) / matrix.size
    return matrix / total


def _probability_dict(probabilities):
    return {
        "home": float(probabilities[0]),
        "draw": float(probabilities[1]),
        "away": float(probabilities[2]),
    }
