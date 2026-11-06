"""Part B smoke-test capture: call API-Football for one league/3-day range and save trimmed raw JSON.

Never prints, logs or stores the key. The key lives only in memory and in the request header.
"""
import json
import os
import sys
from pathlib import Path

from betbrain.providers import apifootball

DATE_FROM, DATE_TO = "2026-11-06", "2026-11-08"
LEAGUE = "Premier League"
OUT = Path("data/samples")


def trim(obj, max_items=2):
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if isinstance(out.get("response"), list):
        items = out["response"][:max_items]
        trimmed = []
        for it in items:
            it = dict(it)
            if isinstance(it.get("bookmakers"), list):      # /odds is huge
                it["bookmakers"] = it["bookmakers"][:1]
            trimmed.append(it)
        out["response"] = trimmed
    return out


def scrub(text, key):
    return text.replace(key, "***REDACTED***") if key else text


def field_report(kind, raw):
    """Check the exact fields apifootball.py reads against the real payload."""
    r = (raw.get("response") or [{}])[0]
    checks = {}
    if kind == "fixtures":
        checks = {
            "fixture.id": r.get("fixture", {}).get("id"),
            "fixture.date": r.get("fixture", {}).get("date"),
            "fixture.status.short": (r.get("fixture", {}).get("status") or {}).get("short"),
            "league.name": (r.get("league") or {}).get("name"),
            "teams.home.name": ((r.get("teams") or {}).get("home") or {}).get("name"),
            "teams.away.name": ((r.get("teams") or {}).get("away") or {}).get("name"),
            "goals.home": (r.get("goals") or {}).get("home"),
            "goals.away": (r.get("goals") or {}).get("away"),
        }
    elif kind == "odds":
        b = (r.get("bookmakers") or [{}])[0]
        bets = b.get("bets") or []
        checks = {
            "fixture.id": (r.get("fixture") or {}).get("id"),
            "bookmakers[].name": b.get("name"),
            "bets[].name": [x.get("name") for x in bets],
            "bets[].values[].value/odd": (bets[0].get("values") if bets else None),
        }
    elif kind == "injuries":
        checks = {
            "fixture.id": (r.get("fixture") or {}).get("id"),
            "team.name": (r.get("team") or {}).get("name"),
            "player.name": (r.get("player") or {}).get("name"),
            "player.type": (r.get("player") or {}).get("type"),
        }
    return checks


def main():
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        print("API_FOOTBALL_KEY is not set; cannot run Part B")
        sys.exit(2)

    OUT.mkdir(parents=True, exist_ok=True)
    captured = {}
    original = apifootball.ApiFootball._get

    def recording_get(self, path, **params):
        j = original(self, path, **params)
        captured.setdefault(path, []).append(j)
        return j

    apifootball.ApiFootball._get = recording_get
    try:
        m, o, i = apifootball.fetch(DATE_FROM, DATE_TO, leagues=[LEAGUE])
    finally:
        apifootball.ApiFootball._get = original

    summary = {"date_from": DATE_FROM, "date_to": DATE_TO, "league": LEAGUE,
               "requests": getattr(apifootball, "LAST_REQUESTS", None),
               "fixtures_rows": len(m), "odds_rows": len(o), "injury_rows": len(i),
               "endpoints": {}}
    for path, calls in captured.items():
        kind = {"fixtures": "fixtures", "odds": "odds", "injuries": "injuries"}.get(path.strip("/"))
        if kind is None:
            continue
        sample = trim(calls[0], 2)
        text = scrub(json.dumps(sample, indent=2), key)
        (OUT / f"{kind}.json").write_text(text, encoding="utf-8")
        summary["endpoints"][kind] = {
            "calls": len(calls),
            "response_items_first_call": len(calls[0].get("response") or []),
            "fields": field_report(kind, calls[0]),
        }

    # request count is tracked by the adapter but not retained after fetch; count captured calls
    summary["requests_captured"] = sum(len(v) for v in captured.values())
    (OUT / "_summary.json").write_text(scrub(json.dumps(summary, indent=2, default=str), key),
                                       encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
