from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from update_data import ensure_dataset_current


DATA_PATH = Path("data/data.csv")
DATA_URL = "https://raw.githubusercontent.com/fuadh999/football-data/main/data/data.csv"

REQUIRED_COLUMNS = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def load_match_data(path=DATA_PATH, fallback_url=DATA_URL):
    """Load football results from local CSV, falling back to the existing remote source."""
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

    if "Date" in prepared.columns:
        prepared["_match_date"] = pd.to_datetime(
            prepared["Date"], errors="coerce", dayfirst=True
        )
        prepared = prepared.sort_values(["_match_date"], na_position="first")

    return prepared.reset_index(drop=True)


def get_teams(df):
    return sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))


def league_profile(df):
    home_goals = df["FTHG"].mean()
    away_goals = df["FTAG"].mean()
    avg_goals = (home_goals + away_goals) / 2

    home_wins = (df["FTHG"] > df["FTAG"]).mean()
    draws = (df["FTHG"] == df["FTAG"]).mean()
    away_wins = (df["FTHG"] < df["FTAG"]).mean()

    return {
        "home_goals": float(home_goals),
        "away_goals": float(away_goals),
        "avg_goals": float(avg_goals),
        "home_advantage": float(np.sqrt(home_goals / away_goals)) if away_goals else 1.0,
        "outcome_prior": np.array([home_wins, draws, away_wins], dtype=float),
    }


def _team_matches(df, team):
    return df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]


def team_goal_history(df, team):
    matches = _team_matches(df, team)
    rows = []

    for _, row in matches.iterrows():
        is_home = row["HomeTeam"] == team
        goals_for = row["FTHG"] if is_home else row["FTAG"]
        goals_against = row["FTAG"] if is_home else row["FTHG"]
        rows.append(
            {
                "goals_for": float(goals_for),
                "goals_against": float(goals_against),
                "is_home": is_home,
            }
        )

    return pd.DataFrame(rows)


def long_term_strength(df, team, profile):
    history = team_goal_history(df, team)
    if history.empty:
        return {"attack": 1.0, "defense": 1.0}

    avg_for = history["goals_for"].mean()
    avg_against = history["goals_against"].mean()

    return {
        "attack": _safe_ratio(avg_for, profile["avg_goals"]),
        "defense": _safe_ratio(avg_against, profile["avg_goals"]),
    }


def weighted_recent_form(df, team, profile, last_n=5):
    history = team_goal_history(df, team).tail(last_n)
    if history.empty:
        return {"attack": 1.0, "defense": 1.0, "matches": 0}

    weights = np.arange(1, len(history) + 1, dtype=float)
    avg_for = np.average(history["goals_for"], weights=weights)
    avg_against = np.average(history["goals_against"], weights=weights)

    return {
        "attack": _safe_ratio(avg_for, profile["avg_goals"]),
        "defense": _safe_ratio(avg_against, profile["avg_goals"]),
        "matches": int(len(history)),
    }


def blended_team_strength(df, team, profile, form_weight=0.35, last_n=5):
    long_term = long_term_strength(df, team, profile)
    form = weighted_recent_form(df, team, profile, last_n=last_n)

    return {
        "attack": _blend(long_term["attack"], form["attack"], form_weight),
        "defense": _blend(long_term["defense"], form["defense"], form_weight),
        "long_term": long_term,
        "form": form,
    }


def expected_goals(df, home_team, away_team, form_weight=0.35, last_n=5):
    profile = league_profile(df)
    home = blended_team_strength(df, home_team, profile, form_weight, last_n)
    away = blended_team_strength(df, away_team, profile, form_weight, last_n)

    home_advantage = profile["home_advantage"]
    home_xg = profile["avg_goals"] * home["attack"] * away["defense"] * home_advantage
    away_xg = profile["avg_goals"] * away["attack"] * home["defense"] / home_advantage

    return {
        "home_xg": float(np.clip(home_xg, 0.15, 4.75)),
        "away_xg": float(np.clip(away_xg, 0.15, 4.75)),
        "home_strength": home,
        "away_strength": away,
        "league_profile": profile,
    }


def poisson_score_matrix(home_xg, away_xg, max_goals=10):
    goals = np.arange(max_goals + 1)
    home_probs = poisson.pmf(goals, home_xg)
    away_probs = poisson.pmf(goals, away_xg)
    matrix = np.outer(home_probs, away_probs)

    total = matrix.sum()
    if total > 0:
        matrix = matrix / total

    return matrix


def outcome_probabilities(score_matrix):
    home_win = np.tril(score_matrix, -1).sum()
    draw = np.diag(score_matrix).sum()
    away_win = np.triu(score_matrix, 1).sum()
    probs = np.array([home_win, draw, away_win], dtype=float)

    return _normalize(probs)


def calibrate_probabilities(probabilities, priors, shrinkage=0.14, max_probability=0.72):
    """Shrink probabilities toward league priors and cap extreme confidence."""
    calibrated = (1 - shrinkage) * np.array(probabilities) + shrinkage * _normalize(priors)
    calibrated = _normalize(calibrated)

    max_idx = int(np.argmax(calibrated))
    if calibrated[max_idx] > max_probability:
        excess = calibrated[max_idx] - max_probability
        calibrated[max_idx] = max_probability

        other_idx = [idx for idx in range(len(calibrated)) if idx != max_idx]
        other_total = calibrated[other_idx].sum()
        if other_total > 0:
            calibrated[other_idx] += excess * calibrated[other_idx] / other_total

    return _normalize(calibrated)


def most_likely_score(score_matrix):
    score = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    return f"{score[0]} - {score[1]}"


def over_under_probability(score_matrix, line=2.5):
    over = 0.0
    for home_goals in range(score_matrix.shape[0]):
        for away_goals in range(score_matrix.shape[1]):
            if home_goals + away_goals > line:
                over += score_matrix[home_goals, away_goals]

    return float(over), float(1 - over)


def predict_match(
    df,
    home_team,
    away_team,
    max_goals=10,
    form_weight=0.35,
    last_n=5,
    calibration_shrinkage=0.14,
):
    xg = expected_goals(df, home_team, away_team, form_weight, last_n)
    matrix = poisson_score_matrix(xg["home_xg"], xg["away_xg"], max_goals=max_goals)

    raw_outcome = outcome_probabilities(matrix)
    calibrated = calibrate_probabilities(
        raw_outcome,
        xg["league_profile"]["outcome_prior"],
        shrinkage=calibration_shrinkage,
    )
    over25, under25 = over_under_probability(matrix, line=2.5)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": xg["home_xg"],
        "away_xg": xg["away_xg"],
        "probabilities": {
            "home_win": float(calibrated[0]),
            "draw": float(calibrated[1]),
            "away_win": float(calibrated[2]),
        },
        "raw_probabilities": {
            "home_win": float(raw_outcome[0]),
            "draw": float(raw_outcome[1]),
            "away_win": float(raw_outcome[2]),
        },
        "score": most_likely_score(matrix),
        "over25": over25,
        "under25": under25,
        "score_matrix": matrix,
        "home_strength": xg["home_strength"],
        "away_strength": xg["away_strength"],
        "home_advantage": xg["league_profile"]["home_advantage"],
    }


def get_odds(df, home_team, away_team):
    odds_columns = ["B365H", "B365D", "B365A"]
    if not set(odds_columns).issubset(df.columns):
        return None, None, None

    match = df[(df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)]
    if match.empty:
        return None, None, None

    row = match.iloc[-1]
    return tuple(pd.to_numeric(row[col], errors="coerce") for col in odds_columns)


def _safe_ratio(value, baseline):
    if baseline is None or baseline == 0 or pd.isna(baseline):
        return 1.0
    return float(value / baseline)


def _blend(base, recent, recent_weight):
    return float((1 - recent_weight) * base + recent_weight * recent)


def _normalize(values):
    arr = np.array(values, dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total
