import pandas as pd

from .config import REPORTS

TITLES = {
    "BANKERS": "Daily Bankers", "TWO_ODDS": "2-Odds Slip", "THREE_ODDS": "3-Odds Slip",
    "OVER_1_5": "Over 1.5 Goals", "STRAIGHT_WINS": "Straight Wins", "MULTI_ACC": "Multi Accumulator",
    "WEEKEND_10": "Weekend ~10 Odds", "WEEKLY_10": "Weekly ~10 Odds",
}


def _pct(x):
    return "n/a" if x is None or x != x else f"{x * 100:.0f}%"


def _leg_rows(legs):
    lines = ["| Date | Fixture | Pick | Odds | AI confidence |", "|---|---|---|---|---|"]
    for l in sorted(legs, key=lambda x: -x["conf"]):
        tag = "" if l["odds_source"] == "book" else " *"
        lines.append(f"| {l['date']} | {l['home']} v {l['away']} | {l['label']} | "
                     f"{l['odds']:.2f}{tag} | {_pct(l['conf'])} |")
    return lines


def render(result, insights=None):
    out = [f"# BetBrain picks for {result['run_date']}", "",
           "_Probabilities are model estimates, not guarantees. `*` = derived/estimated odds: "
           "confirm real prices on your bookmaker before placing._", ""]
    b = result["board"]
    if len(b):
        out += ["## AI ranked board (best pick per match, all sections)", ""]
        out += ["| # | Fixture | Pick | Odds | Confidence |", "|---|---|---|---|---|"]
        for i, r in enumerate(b.itertuples(), 1):
            out.append(f"| {i} | {r.home} v {r.away} | {r.label} | {r.odds:.2f} | {_pct(r.conf)} |")
        out.append("")
    cr = result.get("crowd")
    if cr is not None and len(cr):
        out += ["## Crowd (X / Grok) vs the model", "",
                "_Shown for comparison only. The crowd never changes a pick; the engine records who "
                "was right._", "", "| Fixture | Model | Crowd | Posts | Flag |", "|---|---|---|---|---|"]
        for r in cr.itertuples():
            out.append(f"| {r.home} v {r.away} | {r.model_pick} {r.model_p:.0%} | {r.crowd_pick} "
                       f"{r.crowd_p:.0%} | {r.n_posts} | {r.flag} |")
        cl = result.get("crowd_learn")
        if cl:
            out += ["", f"Crowd track record on {cl['n']} settled matches: crowd picked the result "
                    f"{cl['crowd_acc']:.0%}, model {cl['model_acc']:.0%}. When they disagreed "
                    f"({cl['n_disagree']}x): crowd right "
                    f"{'n/a' if cl['crowd_right_when_disagree'] is None else format(cl['crowd_right_when_disagree'], '.0%')}"
                    f", model right "
                    f"{'n/a' if cl['model_right_when_disagree'] is None else format(cl['model_right_when_disagree'], '.0%')}."]
        out.append("")
    for s in result["slate"] + result["long"]:
        out.append(f"## {TITLES[s['section']]}")
        if not s["legs"]:
            out += ["_No combination met the criteria with today's fixtures._", ""]
            continue
        if s["kind"] == "acca":
            raw = s.get("win_prob_raw")
            extra = f" (model said {_pct(raw)}, cut using past slip results)" if raw and raw - s["win_prob"] > 0.01 else ""
            out.append(f"Total odds **{s['total_odds']:.2f}** | realistic win probability "
                       f"**{_pct(s['win_prob'])}**{extra} | legs {len(s['legs'])}")
        out.append("")
        out += _leg_rows(s["legs"])
        out.append("")
        if s.get("notes"):
            out.append(f"_Why: {s['notes']}_")
        alts = [a for a in s.get("alts", []) if a["n"] != len(s["legs"])]
        if alts:
            out.append("_Other lengths considered: " + "; ".join(
                f"{a['n']} legs -> {a['joint']:.0%} at {a['odds']:.2f}" for a in alts) + "_")
        out.append("")
    if insights and insights.get("n_settled"):
        out += ["## Intelligence engine: what has worked", "",
                f"Learned from {insights['n_settled']} settled predictions.", ""]
        out += ["**What the engine has learned**", ""] + [f"- {r}" for r in result["engine"].rules()] + [""]
        for key, title in (("by_legs", "Win rate by number of legs"), ("by_market", "By market"), ("by_section", "By section (legs)"),
                           ("accas", "Accumulator slips"), ("by_odds_band", "By odds band"),
                           ("calibration", "Confidence vs reality")):
            if key in insights:
                out += [f"**{title}**", "", insights[key].to_markdown(), ""]
        out += [f"Banker thresholds learned: `{insights['banker_thresholds']}` "
                "(None = market never reached the target hit rate, barred from Bankers)", ""]
    return "\n".join(out)


def write(result, insights=None):
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / f"{result['run_date']}.md"
    p.write_text(render(result, insights))
    return p
