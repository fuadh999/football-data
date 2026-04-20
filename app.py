from flask import Flask, request, render_template_string

from decision import build_decision, outcome_label
from model import get_teams, load_match_data, predict_match
from odds import get_match_odds
from team_aliases import TEAM_ALIASES, canonical_team_name, normalize_team_name


app = Flask(__name__)

df = load_match_data()
teams = get_teams(df)
history = []


HTML = """
<h2>Football Value Betting Decision Support</h2>

<form method="post">
{% if error %}
<p style="color:red;"><b>{{error}}</b></p>
{% endif %}

Home Team:<br>
<input list="teams" id="home" name="home">
<br><br>

Away Team:<br>
<input list="teams" id="away" name="away">
<br><br>

<datalist id="teams">
{% for team in teams %}
<option value="{{team}}">
{% endfor %}
</datalist>

<input type="submit" value="Analyze">
</form>

<script>
const teams = {{ teams|tojson }};
const aliases = {{ aliases|tojson }};

function autoSelect(inputId){
    const input = document.getElementById(inputId);
    input.addEventListener("blur", function(){
        let val = input.value.toLowerCase();
        if(val==="") return;
        if(aliases[val]){
            input.value = aliases[val];
            return;
        }
        let matches = teams.filter(t => t.toLowerCase().startsWith(val));
        if(matches.length>0){
            input.value = matches[0];
        }
    });
}

autoSelect("home");
autoSelect("away");
</script>

{% if r %}
<hr>
<h3>Decision</h3>

<p><b>Recommendation:</b> {{r.recommendation}}</p>
<p><b>Selected Market:</b> {{r.selected_label}}</p>
<p><b>Confidence:</b> {{r.confidence_percent}}% ({{r.confidence_label}})</p>
<p><b>Risk:</b> {{r.risk_level}} | Stake: {{r.stake_percent}}% bankroll ({{r.stake_units}} units)</p>
<p><b>Reason:</b> {{r.reason}}</p>

<p><b>O/U 2.5 Recommendation:</b> {{r.over_under_recommendation}}</p>
<p><b>O/U 2.5 Selected Market:</b> {{r.over_under_selected_label}}</p>
<p><b>O/U 2.5 Confidence:</b> {{r.over_under_confidence_percent}}% ({{r.over_under_confidence_label}})</p>
<p><b>O/U 2.5 Risk:</b> {{r.over_under_risk_level}} | Stake: {{r.over_under_stake_percent}}% bankroll ({{r.over_under_stake_units}} units)</p>
<p><b>O/U 2.5 Reason:</b> {{r.over_under_reason}}</p>

<h3>Model Output</h3>
<p><b>xG:</b> {{r.home_team}} {{r.home_xg}} - {{r.away_xg}} {{r.away_team}}</p>

<p><b>Probabilities:</b><br>
{{r.home_team}}: {{r.prob.home}}% |
Draw: {{r.prob.draw}}% |
{{r.away_team}}: {{r.prob.away}}%</p>

<p><b>Under/Over {{r.over_under.line}}:</b><br>
Under: {{r.over_under.under}}% |
Over: {{r.over_under.over}}% |
Prediction: {{r.over_under.prediction}}</p>

<p><b>Predicted Score:</b> {{r.predicted_score}}</p>
<p><b>Highest Probability Outcome:</b> {{r.highest_outcome}}</p>
<p><b>Score Outcome:</b> {{r.score_outcome}}</p>
<p><b>Model Agreement:</b> {{r.agreement}}</p>

<h3>Odds & Value</h3>
<table border="1" cellpadding="5">
<tr>
<th>Market</th>
<th>Odds</th>
<th>Implied Prob</th>
<th>Model Prob</th>
<th>EV</th>
</tr>
{% for row in r.value_rows %}
<tr>
<td>{{row.label}}</td>
<td>{{row.odd}}</td>
<td>{{row.implied_probability}}</td>
<td>{{row.probability}}</td>
<td>{{row.ev}}</td>
</tr>
{% endfor %}
</table>

<h3>Odds & Value O/U 2.5</h3>
<table border="1" cellpadding="5">
<tr>
<th>Market</th>
<th>Odds</th>
<th>Implied Prob</th>
<th>Model Prob</th>
<th>EV</th>
</tr>
{% for row in r.over_under_value_rows %}
<tr>
<td>{{row.label}}</td>
<td>{{row.odd}}</td>
<td>{{row.implied_probability}}</td>
<td>{{row.probability}}</td>
<td>{{row.ev}}</td>
</tr>
{% endfor %}
</table>

<h3>Score Distribution 0-5 (%)</h3>
<table border="1" cellpadding="5">
<tr>
<th>{{r.home_team}} \\ {{r.away_team}}</th>
{% for g in r.goals %}
<th>{{g}}</th>
{% endfor %}
</tr>
{% for row in r.score_matrix %}
<tr>
<th>{{loop.index0}}</th>
{% for cell in row %}
<td>{{cell}}</td>
{% endfor %}
</tr>
{% endfor %}
</table>

{% endif %}

{% if history %}
<hr>
<h3>History</h3>
<table border="1" cellpadding="5">
<tr>
<th>Match</th>
<th>xG</th>
<th>Prob H/D/A</th>
<th>U/O 2.5</th>
<th>Score</th>
<th>Recommendation</th>
<th>Confidence</th>
</tr>
{% for h in history %}
<tr>
<td>{{h.home}} vs {{h.away}}</td>
<td>{{h.home_xg}} - {{h.away_xg}}</td>
<td>{{h.home_prob}} / {{h.draw_prob}} / {{h.away_prob}}</td>
<td>{{h.under25_prob}} / {{h.over25_prob}} ({{h.over_under_prediction}})</td>
<td>{{h.score}}</td>
<td>{{h.recommendation}}</td>
<td>{{h.confidence}}</td>
</tr>
{% endfor %}
</table>
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    global history

    result = None
    error = None

    if request.method == "GET":
        history = []

    if request.method == "POST":
        home_input = request.form.get("home", "").strip()
        away_input = request.form.get("away", "").strip()
        home = resolve_team_name(home_input)
        away = resolve_team_name(away_input)

        if home not in teams:
            error = f"Nama tim '{home_input}' tidak tersedia"
        elif away not in teams:
            error = f"Nama tim '{away_input}' tidak tersedia"
        elif home == away:
            error = "Home dan Away tidak boleh sama"
        else:
            prediction = predict_match(df, home, away, max_goals=5)
            odds = get_match_odds(df, home, away)
            decision = build_decision(prediction, odds)
            result = format_result(prediction, odds, decision)
            history.insert(0, history_item(result))

    return render_template_string(
        HTML,
        teams=teams,
        aliases=TEAM_ALIASES,
        r=result,
        error=error,
        history=history,
    )


def format_result(prediction, odds, decision):
    probabilities = prediction["probabilities"]
    agreement = decision["model_agreement"]
    over_under = prediction["over_under"]
    over_under_decision = decision["over_under"]

    return {
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],
        "home_xg": round(prediction["home_xg"], 2),
        "away_xg": round(prediction["away_xg"], 2),
        "prob": {
            "home": percent(probabilities["home"]),
            "draw": percent(probabilities["draw"]),
            "away": percent(probabilities["away"]),
        },
        "over_under": {
            "line": over_under["line"],
            "under": percent(over_under["under"]),
            "over": percent(over_under["over"]),
            "prediction": over_under_label(over_under["prediction"], over_under["line"]),
        },
        "predicted_score": prediction["predicted_score"],
        "highest_outcome": outcome_label(agreement["highest_probability_outcome"]),
        "score_outcome": outcome_label(agreement["score_outcome"]),
        "agreement": "YES" if agreement["agreement"] else "NO",
        "recommendation": decision["recommendation"],
        "selected_label": decision["selected_label"],
        "confidence_percent": percent(decision["confidence"]),
        "confidence_label": decision["confidence_label"],
        "risk_level": decision["risk"]["risk_level"],
        "stake_percent": round(decision["risk"]["stake_percent"], 2),
        "stake_units": round(decision["risk"]["stake_units"], 2),
        "reason": decision["reason"],
        "over_under_recommendation": over_under_decision["recommendation"],
        "over_under_selected_label": over_under_decision["selected_label"],
        "over_under_confidence_percent": percent(over_under_decision["confidence"]),
        "over_under_confidence_label": over_under_decision["confidence_label"],
        "over_under_risk_level": over_under_decision["risk"]["risk_level"],
        "over_under_stake_percent": round(over_under_decision["risk"]["stake_percent"], 2),
        "over_under_stake_units": round(over_under_decision["risk"]["stake_units"], 2),
        "over_under_reason": over_under_decision["reason"],
        "value_rows": value_rows(decision["values"], odds["match_result"], market_type="match_result"),
        "over_under_value_rows": value_rows(
            decision["over_under_values"],
            odds["over_under_2_5"],
            market_type="over_under_2_5",
        ),
        "score_matrix": prediction["score_matrix_percent"].tolist(),
        "goals": list(range(prediction["score_matrix"].shape[0])),
    }


def resolve_team_name(value):
    cleaned = value.strip()
    if cleaned in teams:
        return cleaned

    canonical = canonical_team_name(cleaned, known_teams=teams)
    if canonical in teams:
        return canonical

    key = normalize_team_name(cleaned)
    lower_map = {normalize_team_name(team): team for team in teams}
    if key in lower_map:
        return lower_map[key]

    startswith_matches = [
        team for team in teams if normalize_team_name(team).startswith(key)
    ]
    if startswith_matches:
        return startswith_matches[0]

    contains_matches = [
        team for team in teams if key and key in normalize_team_name(team)
    ]
    if contains_matches:
        return contains_matches[0]

    return cleaned


def value_rows(values, odds, market_type="match_result"):
    labels = {
        "match_result": {
            "home": "Home Win",
            "draw": "Draw",
            "away": "Away Win",
        },
        "over_under_2_5": {
            "over": "Over 2.5",
            "under": "Under 2.5",
        },
    }
    outcomes = tuple(values.keys())
    rows = []
    for outcome in outcomes:
        item = values[outcome]
        rows.append(
            {
                "label": labels[market_type][outcome],
                "odd": display_number(odds[outcome]),
                "implied_probability": display_percent(item["implied_probability"]),
                "probability": display_percent(item["probability"]),
                "ev": display_number(item["ev"]),
            }
        )
    return rows


def history_item(result):
    return {
        "home": result["home_team"],
        "away": result["away_team"],
        "home_xg": result["home_xg"],
        "away_xg": result["away_xg"],
        "home_prob": result["prob"]["home"],
        "draw_prob": result["prob"]["draw"],
        "away_prob": result["prob"]["away"],
        "under25_prob": result["over_under"]["under"],
        "over25_prob": result["over_under"]["over"],
        "over_under_prediction": result["over_under"]["prediction"],
        "score": result["predicted_score"],
        "recommendation": result["recommendation"],
        "confidence": f'{result["confidence_percent"]}% {result["confidence_label"]}',
    }


def percent(value):
    return round(value * 100, 2)


def display_percent(value):
    if value is None:
        return "-"
    return f"{percent(value)}%"


def display_number(value):
    if value is None:
        return "-"
    return round(float(value), 3)


def over_under_label(prediction, line):
    side = "Over" if prediction == "over" else "Under"
    return f"{side} {line}"


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
