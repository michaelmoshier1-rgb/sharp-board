#!/usr/bin/env python3
"""
Sharp Board  -  NFL · NBA · College Football · College Basketball · NHL
=======================================================================
Predicts the spread (puck line), total and moneyline for every game and compares them with DraftKings' lines.

What goes into each predicted line
  * Offense and defense ratings for every team, opponent-adjusted (ridge regression), recent games
    weighted more, blowouts capped, last season's ratings as the starting point:
        basketball - points per possession (pace-neutral) from ESPN box scores, plus team pace
        football   - points blended with yards-per-play (converted to points) from ESPN box scores
  * Home-court/field edge custom to each team's own building (shrunk toward league average), neutral sites = 0
  * Rest (back-to-backs, 3-in-4 nights / byes, short weeks) and travel distance
  * Key injuries from ESPN's injury report x each player's value (basketball: Game Score above replacement;
    college football: starting QB value; NFL: nfelo's QB model - see below)
  * NFL: nfelo QB-adjusted Elo (github.com/greerreNFL/nfeloqb, updated Tue/Thu) with the projected starting QB,
    blended 50/50 with our own ratings for the spread / moneyline
  * Outdoor football weather (Open-Meteo forecast): wind and rain lower the total and tighten the spread
  * Win % = normal curve on the predicted margin, spread of outcomes (SD) fitted on this season's games

Board highlights (sport tabs): a box is gold when the CURRENT line has value vs. our predicted line,
re-checked every run until kickoff.

Tracked bets (Results tab) LOCK at fixed times at the CURRENT DraftKings line at that moment:
  NFL Tue 9am ET (the week's games) · NCAAF Sun 9am ET (next 7 days) · NBA / NCAAB / NHL daily 9am ET.
Value rules (same for the board and the locks):
  spread: our line beats the opening spread by SPREAD_EDGE points or more
  total : our total is TOTAL_EDGE points or more away from the opening total
  ML    : our win % x opening-price payout - 1 >= ML_EDGE
  Gaps more than MAX_GAP_X x the threshold are treated as missing information and skipped.
  NHL   : puck line, total and ML each need a 5% model edge (Poisson goal model incl. overtime)
Tracked bets are 1u at the locked line/price, written once to board_data.json, graded once from the final
score, and never recalculated.

Run:  python3 sharp_board.py      Flags: --offline  --no-open  --now 2026-11-26T08:00
Standard library only (Python 3.9+).
"""
import csv, io, json, math, os, re, sys, time, urllib.request, webbrowser
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone, date
from statistics import NormalDist

# ================================ CONFIG ====================================
SPORTS = {
    "nfl": dict(label="NFL", kind="fball", path="football/nfl", q="", weekly=True, lock=("tue", 9, 0),
                season="2026-09-08", prev=("2025-09-04", "2026-01-05"),
                sd=13.5, hca=1.7, half_life=70, prior_games=4, regress=0.4, unknown_pct=None, cap=24, hfa_k=16,
                bye=0.8, short_week=-0.8, travel=0.25, qb_base=3.0, qb_slope=1.2, qb_cap=8.0,
                spread_edge=2.0, total_edge=3.0),
    "nba": dict(label="NBA", kind="bball", path="basketball/nba", q="", weekly=False, lock=("daily", 9, 0),
                season="2026-10-20", prev=("2025-10-21", "2026-04-13"),
                sd=12.0, hca=2.4, half_life=40, prior_games=10, regress=0.4, unknown_pct=None, cap=30, hfa_k=40,
                b2b=-1.6, three_in_four=-0.6, travel=0.35, inj_scale=0.45, inj_cap=7.0, team_inj_cap=11.0,
                spread_edge=3.0, total_edge=5.0),
    "cfb": dict(label="NCAAF", kind="fball", path="football/college-football", q="&groups=80&limit=500", weekly=True,
                lock=("sun", 9, 0),
                season="2026-08-27", prev=("2025-08-23", "2026-01-21"),
                sd=15.5, hca=2.6, half_life=60, prior_games=3, regress=0.5, unknown_pct=5, cap=35, hfa_k=12,
                bye=0.5, short_week=-0.5, travel=0.25, qb_base=3.0, qb_slope=1.0, qb_cap=7.0,
                spread_edge=3.5, total_edge=4.5),
    "cbb": dict(label="NCAAB", kind="bball", path="basketball/mens-college-basketball", q="&groups=50&limit=500",
                weekly=False, lock=("daily", 9, 0), season="2026-11-02", prev=("2025-11-03", "2026-04-07"),
                sd=10.5, hca=3.2, half_life=40, prior_games=6, regress=0.55, unknown_pct=8, cap=30, hfa_k=20,
                b2b=-0.8, three_in_four=-0.4, travel=0.35, inj_scale=0.40, inj_cap=6.0, team_inj_cap=9.0,
                spread_edge=3.0, total_edge=4.5),
    "nhl": dict(label="NHL", kind="hockey", path="hockey/nhl", q="&limit=100", weekly=False, lock=("daily", 9, 0),
                season="2026-10-06", prev=("2025-10-07", "2026-04-17"),
                sd=2.3, hca=0.2, half_life=45, prior_games=10, regress=0.4, unknown_pct=None, cap=4, hfa_k=40,
                b2b=-0.15, three_in_four=-0.05, travel=0.03, team_inj_cap=0.6,
                ev_edge=0.05),               # hockey: puck line / total / ML all need a 5% model edge
}
LOCK_GRACE_MIN = 60         # tracked bets lock at the first run within 60 min after the lock time
TRACK_VERSION = 2           # v2 = bets locked at fixed times at the current line (v1 used opening lines)
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
ML_EDGE = 0.05              # moneyline value: our win% x opening payout - 1 >= 5%
ML_MIN_P = 0.15             # never bet a team we give under 15%
MAX_GAP_X = 3.0             # gaps bigger than 3x the threshold = missing info (injury/news) -> skipped
NFELO_URL = "https://raw.githubusercontent.com/greerreNFL/nfeloqb/main/qb_elos.csv"
NFELO_WEIGHT = 0.5          # NFL spread/ML: 50% nfelo QB-adjusted Elo, 50% our ratings
ELO_PER_POINT = 25.0
NFELO_TEAMS = {"OAK": "LV", "SD": "LAC", "STL": "LAR", "WAS": "WSH", "JAC": "JAX", "LA": "LAR"}
YPP_BLEND = 0.5             # football: 50% actual points, 50% yards-based points
WIND_TOTAL = ((25, -5.0), (18, -3.0), (13, -1.5))   # (mph at kickoff, total adjustment)
RAIN_TOTAL = (2.0, -2.0)                           # (mm/h, total adjustment)
WEATHER_SPREAD_SHRINK = 0.95
INJ_STATUS = {"out": 1.0, "injured reserve": 1.0, "suspension": 1.0, "doubtful": 0.75,
              "questionable": 0.35, "day-to-day": 0.3}
MIN_PRIOR_GAMES = 10
MIN_NEW_TEAM_GAMES = 4
MIN_BT_GAMES = 150
FINAL_AFTER_H = 2.0
WORKERS = 12
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "board_data.json")
CACHE_DIR = os.path.join(HERE, "cache")
OUT_HTML = os.path.join(HERE, "sharp_board.html")
TEMPLATE = os.path.join(HERE, "board_template.html")
SB = "https://site.api.espn.com/apis/site/v2/sports/{}/scoreboard?dates={}{}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/{}/summary?event={}"
GEO = "https://geocoding-api.open-meteo.com/v1/search?name={}&count=10&language=en&format=json"
WX = ("https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&hourly=wind_speed_10m,precipitation"
      "&wind_speed_unit=mph&timezone=UTC&start_date={}&end_date={}")
N = NormalDist()
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-5))


# ------------------------------- odds / ESPN parsing ------------------------
def line_num(x):
    s = str(x if x is not None else "").strip().lower().lstrip("ou")
    if s in ("pk", "pick", "even", "ev"):
        return 0.0
    try:
        return float(s.replace("+", ""))
    except Exception:
        return None


def odds_full(lst):
    """DraftKings spread / total / moneyline, open and current, from an ESPN odds or pickcenter list."""
    if not lst:
        return None
    o = next((x for x in lst if "draft" in str(dget(x, "provider", "name") or "").lower()), lst[0])
    out = {}
    def pair(which):
        l = line_num(dget(o, "pointSpread", "home", which, "line"))
        if l is None:
            al = line_num(dget(o, "pointSpread", "away", which, "line"))
            l = -al if al is not None else None
        if l is None:
            return None
        return [l, to_num(dget(o, "pointSpread", "home", which, "odds")) or -110,
                to_num(dget(o, "pointSpread", "away", which, "odds")) or -110]
    so, sc = pair("open"), pair("close")
    if sc is None and o.get("spread") is not None and o.get("details"):
        det = str(o["details"]).split()
        v = line_num(det[-1]) if det else None
        home_ab = str(dget(o, "homeTeamOdds", "team", "abbreviation") or "").upper()
        if v is not None:
            sc = [v if det[0].upper() == home_ab else -v,
                  to_num(dget(o, "homeTeamOdds", "spreadOdds")) or -110, to_num(dget(o, "awayTeamOdds", "spreadOdds")) or -110]
    if so or sc:
        out["spread"] = dict(open=so or sc, cur=sc or so)
    def tot(which):
        t = line_num(dget(o, "total", "over", which, "line"))
        if t is None:
            return None
        return [t, to_num(dget(o, "total", "over", which, "odds")) or -110, to_num(dget(o, "total", "under", which, "odds")) or -110]
    to_, tc = tot("open"), tot("close")
    if tc is None and to_num(o.get("overUnder")):
        tc = [to_num(o.get("overUnder")), to_num(o.get("overOdds")) or -110, to_num(o.get("underOdds")) or -110]
    if to_ or tc:
        out["total"] = dict(open=to_ or tc, cur=tc or to_)
    def ml(which):
        h, a = to_num(dget(o, "moneyline", "home", which, "odds")), to_num(dget(o, "moneyline", "away", which, "odds"))
        if which == "close" and (h is None or a is None):
            h = h if h is not None else to_num(dget(o, "homeTeamOdds", "moneyLine"))
            a = a if a is not None else to_num(dget(o, "awayTeamOdds", "moneyLine"))
        return [h, a] if h and a else None
    mo, mc = ml("open"), ml("close")
    if mo or mc:
        out["ml"] = dict(open=mo or mc, cur=mc or mo)
    return out or None


def dk_obj(lst):  # used by the shared box-score parser
    return odds_full(lst)


def team_info(c):
    t = c.get("team") or {}
    rec = next((r.get("summary") for r in (c.get("records") or []) if r.get("type") in ("total", None) or r.get("name") == "overall"), None)
    return dict(id=str(t.get("id") or c.get("id") or ""), abbr=(t.get("abbreviation") or "").upper(),
                location=t.get("location") or "", name=t.get("displayName") or "", short=t.get("shortDisplayName") or "",
                logo=t.get("logo") or "", rec=rec or "", rank=dget(c, "curatedRank", "current"))


def parse_event(ev):
    stype = dget(ev, "season", "type")
    if stype is not None and stype not in (2, 3):
        return None
    comp = (ev.get("competitions") or [{}])[0]
    cs = comp.get("competitors") or []
    H = next((c for c in cs if c.get("homeAway") == "home"), None)
    A = next((c for c in cs if c.get("homeAway") == "away"), None)
    if not H or not A:
        return None
    st = (comp.get("status") or ev.get("status") or {}).get("type") or {}
    name = str(st.get("name") or "")
    status = "final" if st.get("completed") else ("off" if ("POSTPONED" in name or "CANCEL" in name)
                                                  else ("live" if st.get("state") == "in" else "pre"))
    v = comp.get("venue") or {}
    ad = v.get("address") or {}
    return dict(id=str(ev["id"]), commence=parse_t(ev["date"]).isoformat(), home=team_info(H), away=team_info(A),
                neutral=bool(comp.get("neutralSite")), city=ad.get("city") or "", state=ad.get("state") or ad.get("country") or "",
                indoor=bool(v.get("indoor")), venue=v.get("fullName") or "", status=status,
                detail=st.get("shortDetail") or "", hs=to_num(H.get("score")), as_=to_num(A.get("score")),
                odds=odds_full(comp.get("odds")))


# ------------------------------- cache --------------------------------------
def cache_path(sk):
    return os.path.join(CACHE_DIR, f"{sk}.json")


def load_cache(sk):
    c = {}
    if os.path.exists(cache_path(sk)):
        with open(cache_path(sk)) as f:
            c = json.load(f)
    for k, v in (("games", {}), ("done", []), ("box", {}), ("prior", None), ("bt", None)):
        c.setdefault(k, v)
    return c


def save_cache(sk, c):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = cache_path(sk) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f, separators=(",", ":"))
    os.replace(tmp, cache_path(sk))


def fetch_dates(cfg, dates):
    res = pmap(lambda d: FETCH(SB.format(cfg["path"], d.strftime("%Y%m%d"), cfg["q"])), dates)
    return {d: [g for g in (parse_event(e) for e in data.get("events", [])) if g] for d, data in res.items() if data is not None}


def window(cfg, now):
    """The slate shown on the sport's tab: today (daily) or Tue->Mon week (weekly)."""
    if cfg["weekly"]:
        start = (now - timedelta(days=(now.weekday() - 1) % 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=7)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def refresh_games(sk, cfg, c, now, notes):
    start = date.fromisoformat(cfg["season"])
    a, b = window(cfg, now)
    end = (b - timedelta(seconds=1)).date()
    if cfg["weekly"]:                                   # weekly locks need the next 7 days of games
        end = max(end, now.date() + timedelta(days=8))
    recent = now.date() - timedelta(days=3)
    need = [d for d in date_range(start, end) if d >= recent or d.isoformat() not in c["done"]]
    got = fetch_dates(cfg, need)
    if need and not got:
        notes.append(f"{cfg['label']}: ESPN unavailable this run.")
    done = set(c["done"])
    for d, games in got.items():
        for g in games:
            old = c["games"].get(g["id"])
            if old:
                if not g.get("odds") and old.get("odds"):
                    g["odds"] = old["odds"]
                elif g.get("odds") and old.get("odds"):
                    for mk in ("spread", "total", "ml"):        # the FIRST opening line we saw is kept
                        if mk in old["odds"] and mk in g["odds"]:
                            g["odds"][mk]["open"] = old["odds"][mk]["open"]
                        elif mk in old["odds"]:
                            g["odds"][mk] = old["odds"][mk]
            c["games"][g["id"]] = g
        if d < now.date() - timedelta(days=1) and all(g["status"] in ("final", "off") for g in games):
            done.add(d.isoformat())
    c["done"] = sorted(done)


def refresh_boxes(sk, cfg, c, limit=6000):
    need = [gid for gid, g in c["games"].items() if g["status"] == "final" and gid not in c["box"]][:limit]
    res = pmap(lambda gid: FETCH(SUMMARY.format(cfg["path"], gid)), need)
    for gid, sm in res.items():
        if sm:
            c["box"][gid] = parse_box(sm, cfg["kind"])


def build_prior(sk, cfg, c, notes):
    if c["prior"]:
        return
    a, b = (date.fromisoformat(x) for x in cfg["prev"])
    got = fetch_dates(cfg, list(date_range(a, b)))
    obs = []
    for games in got.values():
        for g in games:
            if g["status"] == "final" and g["hs"] is not None:
                obs.append((g["commence"], g["home"]["id"], g["away"]["id"], g["neutral"], g["hs"], g["as_"], 1.0))
    cnt = Counter(x for o in obs for x in (o[1], o[2]))
    obs = [o for o in obs if cnt[o[1]] >= MIN_PRIOR_GAMES and cnt[o[2]] >= MIN_PRIOR_GAMES]
    if len(obs) < 50:
        notes.append(f"{cfg['label']}: last season unavailable; ratings start at average.")
        c["prior"] = {"o": {}, "d": {}, "hfa": {}, "mu": None, "unk": 0.0, "n": len(obs)}
        return
    F = fit_od(obs, {}, {}, None, 0.0, cfg["hca"], lam=1.0, cap=cfg["cap"])
    k = 1 - cfg["regress"]
    net = {t: F["o"][t] + F["d"][t] for t in F["o"]}
    unk = pct_rank(list(net.values()), cfg["unknown_pct"]) * k / 2 if cfg["unknown_pct"] else 0.0
    c["prior"] = {"o": {t: round(v * k, 3) for t, v in F["o"].items()}, "d": {t: round(v * k, 3) for t, v in F["d"].items()},
                  "hfa": F["hfa_res"], "mu": F["mu"], "unk": round(unk, 3), "n": len(obs)}


# ------------------------------- ratings ------------------------------------
def fit_od(obs, po, pd, mu0, unk, hca0, lam, half_life=None, asof=None, cap=40, iters=60, hfa_prior=None):
    """obs: (time, home, away, neutral, home_pts, away_pts, w). Offense/defense ridge:
       home_pts = mu + o_h - d_a + hca/2,  away_pts = mu + o_a - d_h - hca/2."""
    if asof is not None and half_life:
        obs = [(t, h, a, n, ph, pa, w * 0.5 ** (max(0, (asof - parse_t(t)).days) / half_life)) for t, h, a, n, ph, pa, w in obs]
    teams = {x for o in obs for x in (o[1], o[2])}
    if not obs:
        return dict(o={}, d={}, mu=mu0 or 0.0, hca=hca0, hfa_res={}, n=Counter())
    W = sum(o[6] for o in obs)
    mu = sum(o[6] * (o[4] + o[5]) / 2 for o in obs) / W
    o_ = {t: po.get(t, unk) for t in teams}
    d_ = {t: pd.get(t, unk) for t in teams}
    hca = hca0
    for _ in range(iters):
        num = den = 0.0
        for t, h, a, n, ph, pa, w in obs:
            if not n:
                num += w * ((ph - pa) - ((o_[h] - d_[a]) - (o_[a] - d_[h])))
                den += w
        hca = (num + 1500 * hca0) / (den + 1500)
        So, Sd, Wt = defaultdict(float), defaultdict(float), defaultdict(float)
        for t, h, a, n, ph, pa, w in obs:
            e = 0.0 if n else hca / 2
            So[h] += w * (ph - mu + d_[a] - e); Sd[a] += w * (mu + o_[h] + e - ph)
            So[a] += w * (pa - mu + d_[h] + e); Sd[h] += w * (mu + o_[a] - e - pa)
            Wt[h] += w; Wt[a] += w
        for k in teams:
            o_[k] = 0.5 * o_[k] + 0.5 * (So[k] + lam * po.get(k, unk)) / (Wt[k] + lam)
            d_[k] = 0.5 * d_[k] + 0.5 * (Sd[k] + lam * pd.get(k, unk)) / (Wt[k] + lam)
    # each building's own edge: home-margin residuals vs league home edge
    res = defaultdict(lambda: [0.0, 0.0])
    for t, h, a, n, ph, pa, w in obs:
        if not n:
            r = (ph - pa) - ((o_[h] - d_[a]) - (o_[a] - d_[h]) + hca)
            res[h][0] += w * r; res[h][1] += w
    n_games = Counter(x for o in obs for x in (o[1], o[2]))
    return dict(o=o_, d=d_, mu=mu, hca=hca, hfa_res={t: [round(v[0], 3), round(v[1], 3)] for t, v in res.items()}, n=n_games)


def game_points(g, box, cfg, lp, pts_per_yd):
    """Adjusted points for (home, away): pace-neutral for basketball, points/yards blend for football."""
    hs, as_ = g["hs"], g["as_"]
    tb = (box or {}).get("teams") or {}
    H, A = tb.get(g["home"]["id"]), tb.get(g["away"]["id"])
    if cfg["kind"] == "bball" and H and A and H.get("poss") and A.get("poss") and lp:
        poss = (H["poss"] + A["poss"]) / 2
        hs, as_ = hs / poss * lp, as_ / poss * lp
    elif cfg["kind"] == "fball" and H and A and H.get("yds") is not None and A.get("yds") is not None and pts_per_yd:
        hs = (1 - YPP_BLEND) * hs + YPP_BLEND * H["yds"] * pts_per_yd
        as_ = (1 - YPP_BLEND) * as_ + YPP_BLEND * A["yds"] * pts_per_yd
    mid = (hs + as_) / 2
    m = max(-cfg["cap"], min(cfg["cap"], hs - as_))
    return mid + m / 2, mid - m / 2


def league_numbers(c):
    ps = [t["poss"] for b in c["box"].values() for t in (b.get("teams") or {}).values() if t.get("poss")]
    lp = sum(ps) / len(ps) if ps else None
    pts = yds = 0.0
    for gid, b in c["box"].items():
        g = c["games"].get(gid)
        if not g or g["hs"] is None:
            continue
        tb = b.get("teams") or {}
        H, A = tb.get(g["home"]["id"]), tb.get(g["away"]["id"])
        if H and A and H.get("yds") is not None and A.get("yds") is not None:
            pts += g["hs"] + g["as_"]; yds += H["yds"] + A["yds"]
    return lp, (pts / yds if yds else None)


def fit(c, cfg, feats, asof, before=None):
    prior = c["prior"] or {}
    po, pd, unk = prior.get("o", {}), prior.get("d", {}), prior.get("unk", 0.0)
    cnt = Counter(x for g in c["games"].values() if g["status"] == "final" and (before is None or parse_t(g["commence"]) < before)
                  for x in (g["home"]["id"], g["away"]["id"]))
    ok = lambda t: t in po or cnt[t] >= MIN_NEW_TEAM_GAMES
    lp, ppy = league_numbers(c)
    obs, obs_p = [], []
    for g in c["games"].values():
        if g["status"] != "final" or g["hs"] is None or (before and parse_t(g["commence"]) >= before):
            continue
        if not (ok(g["home"]["id"]) and ok(g["away"]["id"])):
            continue
        box = c["box"].get(g["id"])
        ph, pa = game_points(g, box, cfg, lp, ppy)
        s = feats.get(g["id"], (0.0, []))[0]
        obs.append((g["commence"], g["home"]["id"], g["away"]["id"], g["neutral"], ph - s / 2, pa + s / 2, 1.0))
        tb = (box or {}).get("teams") or {}
        if cfg["kind"] == "bball" and g["home"]["id"] in tb and g["away"]["id"] in tb:
            obs_p.append((g["home"]["id"], g["away"]["id"], (tb[g["home"]["id"]]["poss"] + tb[g["away"]["id"]]["poss"]) / 2))
    F = fit_od(obs, po, pd, prior.get("mu"), unk, cfg["hca"], cfg["prior_games"], cfg["half_life"], asof, cfg["cap"])
    P, pace = pace_ratings(obs_p) if cfg["kind"] == "bball" else (None, {})
    hfa = {}
    for t in set(F["o"]) | set(prior.get("hfa", {})):
        r1, w1 = F["hfa_res"].get(t, [0.0, 0.0])
        r0, w0 = prior.get("hfa", {}).get(t, [0.0, 0.0])
        hfa[t] = F["hca"] + (r1 + 0.5 * r0) / (w1 + 0.5 * w0 + cfg["hfa_k"])
    F.update(P=P, pace=pace, lp=lp, hfa=hfa, po=po, pd=pd, unk=unk, ok=ok)
    return F


def rated(F, tid):
    return tid in F["po"] or F["n"].get(tid, 0) >= MIN_NEW_TEAM_GAMES


def predict(g, F, cfg, feats, inj=(0.0, 0.0), wx=None, elo_margin=None):
    """Predicted home margin and total, with the pieces behind them."""
    o, d, unk = F["o"], F["d"], F["unk"]
    h, a = g["home"]["id"], g["away"]["id"]
    oh, dh = o.get(h, F["po"].get(h, unk)), d.get(h, F["pd"].get(h, unk))
    oa, da = o.get(a, F["po"].get(a, unk)), d.get(a, F["pd"].get(a, unk))
    hfa = 0.0 if g["neutral"] else F["hfa"].get(h, F["hca"])
    sched = feats.get(g["id"], (0.0, []))[0]
    m = (oh - da) - (oa - dh) + hfa + sched
    tot = 2 * F["mu"] + (oh - da) + (oa - dh)
    if cfg["kind"] == "bball" and F["P"] and F["lp"]:
        k = (F["P"] + F["pace"].get(h, 0) + F["pace"].get(a, 0)) / F["lp"]
        m *= k; tot *= k
    ih, ia = inj
    m += ia - ih
    tot -= 0.5 * (ih + ia)
    if elo_margin is not None:
        m = NFELO_WEIGHT * (elo_margin + hfa + sched) + (1 - NFELO_WEIGHT) * m
    wxt = None
    if wx:
        wind, rain = wx
        adj = next((v for mph, v in WIND_TOTAL if wind >= mph), 0.0) + (RAIN_TOTAL[1] if rain >= RAIN_TOTAL[0] else 0.0)
        if adj:
            tot += adj; m *= WEATHER_SPREAD_SHRINK
            wxt = f"wind {wind:.0f} mph{f', rain {rain:.1f} mm/h' if rain >= RAIN_TOTAL[0] else ''} ({adj:+.1f} total)"
    return m, tot, dict(rating=round((oh + dh) - (oa + da), 1), hfa=round(hfa, 1), sched=round(sched, 1),
                        inj=round(ia - ih, 1), wx=wxt, elo=None if elo_margin is None else round(elo_margin, 1))


# ------------------------------- NFL: nfelo QB-adjusted Elo -----------------
NFELO = {"rows": None}


def nfelo_rows():
    if NFELO["rows"] is None:
        try:
            raw = FETCH_TEXT(NFELO_URL)
            NFELO["rows"] = [r for r in csv.DictReader(io.StringIO(raw)) if r.get("season", "") >= "2025"]
        except Exception:
            NFELO["rows"] = []
    return NFELO["rows"]


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sharp-board)"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8")


FETCH_TEXT = fetch_text


def nfelo_margin(g):
    """(home margin in points from nfelo QB-adjusted Elo, text) for an NFL game, or (None, None)."""
    d = parse_t(g["commence"]).date()
    ha, aa = g["home"]["abbr"], g["away"]["abbr"]
    for r in nfelo_rows():
        t1, t2 = NFELO_TEAMS.get(r["team1"], r["team1"]), NFELO_TEAMS.get(r["team2"], r["team2"])
        try:
            rd = date.fromisoformat(r["date"])
        except Exception:
            continue
        if abs((rd - d).days) > 1 or {t1, t2} != {ha, aa}:
            continue
        try:
            e1, e2 = float(r["qbelo1_pre"]), float(r["qbelo2_pre"])
        except Exception:
            return None, None
        m = (e1 - e2) / ELO_PER_POINT
        if t1 != ha:
            m = -m
        qbs = f"QBs: {r.get('qb2') if t1 == ha else r.get('qb1')} @ {r.get('qb1') if t1 == ha else r.get('qb2')}"
        return m, qbs
    return None, None


# ------------------------------- weather ------------------------------------
def weather(g, geo):
    if g.get("indoor"):
        return None
    loc = geo.get(place(g))
    if not loc:
        return None
    t = parse_t(g["commence"]).astimezone(timezone.utc)
    if t - datetime.now(timezone.utc) > timedelta(days=14):
        return None
    try:
        d = FETCH(WX.format(loc[0], loc[1], t.date().isoformat(), t.date().isoformat()))
        hrs = d["hourly"]["time"]
        i = min(range(len(hrs)), key=lambda k: abs((datetime.fromisoformat(hrs[k]).replace(tzinfo=timezone.utc) - t).total_seconds()))
        return (d["hourly"]["wind_speed_10m"][i] or 0.0, d["hourly"]["precipitation"][i] or 0.0)
    except Exception:
        return None


# ------------------------------- shared helpers ----------------------------
STATES = dict(AL="Alabama", AK="Alaska", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado", CT="Connecticut",
              DE="Delaware", DC="District of Columbia", FL="Florida", GA="Georgia", HI="Hawaii", ID="Idaho", IL="Illinois",
              IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana", ME="Maine", MD="Maryland",
              MA="Massachusetts", MI="Michigan", MN="Minnesota", MS="Mississippi", MO="Missouri", MT="Montana",
              NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey", NM="New Mexico", NY="New York",
              NC="North Carolina", ND="North Dakota", OH="Ohio", OK="Oklahoma", OR="Oregon", PA="Pennsylvania",
              RI="Rhode Island", SC="South Carolina", SD="South Dakota", TN="Tennessee", TX="Texas", UT="Utah",
              VT="Vermont", VA="Virginia", WA="Washington", WV="West Virginia", WI="Wisconsin", WY="Wyoming",
              PR="Puerto Rico", ON="Ontario", QC="Quebec", BC="British Columbia", AB="Alberta", MB="Manitoba")


def http_get(url):
    err = ""
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sharp-model)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            err = str(e)
            time.sleep(1.0 * (i + 1))
    raise IOError(err)


FETCH = http_get  # swapped out in tests


def pmap(fn, items):
    out = {}
    items = list(items)
    if not items:
        return out
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception:
                out[futs[f]] = None
    return out


def dget(o, *path):
    for p in path:
        if isinstance(o, list):
            o = o[p] if isinstance(p, int) and len(o) > p else None
        elif isinstance(o, dict):
            o = o.get(p)
        else:
            return None
    return o


def nv_home(h_am, a_am):
    """No-vig home win prob from two American prices."""
    dh, da = dec_from_american(h_am), dec_from_american(a_am)
    if not dh or not da:
        return None
    ih, ia = 1 / dh, 1 / da
    return ih / (ih + ia)


def miles(a, b):
    if not a or not b:
        return None
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(min(1, h)))


def pct_rank(vals, pct):
    v = sorted(vals)
    if not v:
        return 0.0
    return v[min(len(v) - 1, max(0, int(len(v) * pct / 100)))]


def logloss(p, y):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def parse_t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(ET)


def to_num(x):
    try:
        s = str(x).strip().lower().lstrip("ou").replace("+", "")
        if s in ("even", "ev", "pk", "pick"):
            return 100.0 if "ev" in s else 0.0
        return float(s)
    except Exception:
        return None


def dec_from_american(a):
    if a is None or a == 0:
        return None
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def american(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def fmt_american(d):
    a = american(d)
    return f"+{a}" if a > 0 else str(a)


def date_range(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def stat_map(team_block):
    return {s.get("name"): s.get("displayValue") for s in (team_block.get("statistics") or [])}


def split2(s, sep="-"):
    try:
        a, b = str(s).replace("/", "-").split("-")[:2]
        return float(a), float(b)
    except Exception:
        return None, None


def parse_box(sm, kind):
    """Team efficiency stats, player lines, DraftKings open/close from a completed-game summary."""
    out = dict(teams={}, players={}, dk=dk_obj(sm.get("pickcenter")))
    for tb in dget(sm, "boxscore", "teams") or []:
        tid = str(dget(tb, "team", "id"))
        m = stat_map(tb)
        if kind == "bball":
            fgm, fga = split2(m.get("fieldGoalsMade-fieldGoalsAttempted"))
            ftm, fta = split2(m.get("freeThrowsMade-freeThrowsAttempted"))
            orb = to_num(m.get("offensiveRebounds"))
            tov = to_num(m.get("totalTurnovers")) if m.get("totalTurnovers") is not None else to_num(m.get("turnovers"))
            if None not in (fga, fta, orb, tov):
                out["teams"][tid] = dict(poss=round(fga - orb + tov + 0.44 * fta, 2))
        else:
            plays, yds = to_num(m.get("totalOffensivePlays")), to_num(m.get("totalYards"))
            if plays and yds is not None:
                out["teams"][tid] = dict(plays=plays, yds=yds, to=to_num(m.get("turnovers")) or 0)
    for pb in dget(sm, "boxscore", "players") or []:
        tid = str(dget(pb, "team", "id"))
        rows = []
        for cat in pb.get("statistics") or []:
            keys = cat.get("keys") or []
            if kind == "bball":
                if "minutes" not in keys:
                    continue
                ix = {k: i for i, k in enumerate(keys)}
                for a in cat.get("athletes") or []:
                    st = a.get("stats") or []
                    if a.get("didNotPlay") or len(st) < len(keys):
                        continue
                    mn = to_num(st[ix["minutes"]])
                    if not mn:
                        continue
                    g = lambda k: to_num(st[ix[k]]) if k in ix else 0.0
                    fgm, fga = split2(st[ix["fieldGoalsMade-fieldGoalsAttempted"]]) if "fieldGoalsMade-fieldGoalsAttempted" in ix else (0, 0)
                    ftm, fta = split2(st[ix["freeThrowsMade-freeThrowsAttempted"]]) if "freeThrowsMade-freeThrowsAttempted" in ix else (0, 0)
                    gs = ((g("points") or 0) + 0.4 * (fgm or 0) - 0.7 * (fga or 0) - 0.4 * ((fta or 0) - (ftm or 0))
                          + 0.7 * (g("offensiveRebounds") or 0) + 0.3 * (g("defensiveRebounds") or 0) + (g("steals") or 0)
                          + 0.7 * (g("assists") or 0) + 0.7 * (g("blocks") or 0) - 0.4 * (g("fouls") or 0) - (g("turnovers") or 0))
                    rows.append([str(dget(a, "athlete", "id")), dget(a, "athlete", "displayName") or "", mn, round(gs, 2)])
            elif cat.get("name") == "passing":
                ix = {k: i for i, k in enumerate(keys)}
                for a in cat.get("athletes") or []:
                    st = a.get("stats") or []
                    try:
                        comp, att = split2(st[ix["completions/passingAttempts"]])
                        yds = to_num(st[ix["passingYards"]]) or 0
                        td = to_num(st[ix["passingTouchdowns"]]) or 0
                        it = to_num(st[ix["interceptions"]]) or 0
                        sk, sky = split2(st[ix["sacks-sackYardsLost"]]) if "sacks-sackYardsLost" in ix else (0, 0)
                    except Exception:
                        continue
                    if att:
                        rows.append([str(dget(a, "athlete", "id")), dget(a, "athlete", "displayName") or "",
                                     att, yds, td, it, sk or 0, sky or 0])
        if rows:
            out["players"][tid] = rows
    return out


def parse_injuries(sm):
    """{team_id: [(athlete_id, name, status_weight, status_text)]} from a pre-game summary."""
    out = {}
    for tb in sm.get("injuries") or []:
        tid = str(dget(tb, "team", "id"))
        lst = []
        for inj in tb.get("injuries") or []:
            stt = str(inj.get("status") or dget(inj, "type", "description") or "").lower()
            w = next((v for k, v in INJ_STATUS.items() if k in stt), 0.0)
            if not w:
                continue
            a = inj.get("athlete") or {}
            aid = str(a.get("id") or "")
            if not aid:
                for l in a.get("links") or []:
                    m = re.search(r"/id/(\d+)", str(l.get("href") or ""))
                    if m:
                        aid = m.group(1)
                        break
            lst.append((aid, a.get("displayName") or "", w, inj.get("status") or stt))
        out[tid] = lst
    return out


def load_geo():
    p = os.path.join(CACHE_DIR, "geo.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_geo(geo):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "geo.json"), "w") as f:
        json.dump(geo, f, separators=(",", ":"))


def geocode_all(geo, places):
    need = [p for p in places if p not in geo and p.split("|")[0]]
    def one(p):
        city, st = p.split("|")
        data = FETCH(GEO.format(urllib.request.quote(city)))
        res = data.get("results") or []
        full = STATES.get(st, st)
        best = next((r for r in res if r.get("admin1") == full), None) or \
               next((r for r in res if r.get("country_code") in ("US", "CA")), None) or (res[0] if res else None)
        return [round(best["latitude"], 4), round(best["longitude"], 4)] if best else None
    for p, v in pmap(one, need).items():
        geo[p] = v
    return geo


def place(g):
    return f"{g.get('city','')}|{g.get('state','')}"


def schedule_features(c, cfg, geo):
    """Per-game rest and travel adjustments (points to ADD to the home margin) + detail text."""
    games = sorted(c["games"].values(), key=lambda g: g["commence"])
    homes = defaultdict(Counter)
    for g in games:
        if not g["neutral"] and g.get("city"):
            homes[g["home"]["id"]][place(g)] += 1
    home_loc = {t: geo.get(cnt.most_common(1)[0][0]) for t, cnt in homes.items()}
    diffs = []
    trav = {}
    for g in games:
        v = geo.get(place(g))
        mh = 0.0 if not g["neutral"] else (miles(home_loc.get(g["home"]["id"]), v) or 0.0)
        ma = miles(home_loc.get(g["away"]["id"]), v)
        if ma is None:
            continue
        trav[g["id"]] = (mh, ma)
        if not g["neutral"]:
            diffs.append(ma - mh)
    base = sum(diffs) / len(diffs) if diffs else 0.0
    last = defaultdict(list)
    feats = {}
    for g in games:
        t = parse_t(g["commence"])
        adj, det = 0.0, []
        for side, sign in (("home", 1), ("away", -1)):
            tid = g[side]["id"]
            prev = last[tid]
            a = 0.0
            if prev:
                rest = (t.date() - prev[-1].date()).days
                if cfg["kind"] in ("bball", "hockey"):
                    if rest <= 1:
                        a += cfg["b2b"]
                        det.append(f"{g[side]['short'] or g[side]['abbr']} B2B")
                    if sum(1 for p in prev[-3:] if (t.date() - p.date()).days <= 3) >= 2:
                        a += cfg["three_in_four"]
                else:
                    if rest >= 12:
                        a += cfg["bye"]
                        det.append(f"{g[side]['short'] or g[side]['abbr']} off bye")
                    elif rest <= 5:
                        a += cfg["short_week"]
                        det.append(f"{g[side]['short'] or g[side]['abbr']} short week")
            adj += sign * a
            last[tid].append(t)
        if g["id"] in trav:
            mh, ma = trav[g["id"]]
            ta = cfg["travel"] * ((ma - mh) - (0.0 if g["neutral"] else base)) / 1000
            ta = max(-2.0, min(2.0, ta))
            adj += ta
            if abs(ta) >= 0.3:
                det.append(f"travel {'+' if ta > 0 else ''}{ta:.1f}")
        feats[g["id"]] = (round(adj, 3), det)
    return feats


def pace_ratings(obs_p):
    """obs_p: (home, away, poss). poss = P + p_h + p_a."""
    if not obs_p:
        return 0.0, {}
    P = sum(o[2] for o in obs_p) / len(obs_p)
    p = defaultdict(float)
    for _ in range(25):
        S, W = defaultdict(float), defaultdict(float)
        for h, a, x in obs_p:
            S[h] += x - P - p[a]; W[h] += 1
            S[a] += x - P - p[h]; W[a] += 1
        for k in S:
            p[k] = S[k] / (W[k] + 5)
    return P, dict(p)


def player_values(c, cfg):
    """{team_id: {athlete_id: (name, value_pts, share_of_games)}} and a name index."""
    tg = Counter()
    for gid, b in c["box"].items():
        for tid in (b.get("players") or {}):
            tg[tid] += 1
    vals = defaultdict(dict)
    if cfg["kind"] == "bball":
        agg = defaultdict(lambda: [0, 0.0, 0.0, ""])
        for b in c["box"].values():
            for tid, rows in (b.get("players") or {}).items():
                for pid, name, mn, gs in rows:
                    x = agg[(tid, pid)]
                    x[0] += 1; x[1] += mn; x[2] += gs; x[3] = name
        rot = [x[2] / x[1] for x in agg.values() if x[0] >= 3 and x[1] / x[0] >= 12]
        repl = pct_rank(rot, 20) if rot else 0.0
        for (tid, pid), (gp, mn, gs, name) in agg.items():
            if gp < 2:
                continue
            v = max(0.0, (gs / mn - repl) * (mn / gp)) * cfg["inj_scale"]
            vals[tid][pid] = (name, round(min(v, cfg["inj_cap"]), 2), round(gp / max(tg[tid], 1), 3))
    else:
        qbs = defaultdict(lambda: [0, 0.0, 0.0, 0, ""])   # starts, att, value-yards, att+sacks, name
        all_any = [0.0, 0.0]
        for b in c["box"].values():
            for tid, rows in (b.get("players") or {}).items():
                if not rows:
                    continue
                lead = max(rows, key=lambda r: r[2])
                for pid, name, att, yds, td, it, sk, sky in rows:
                    ay = yds + 20 * td - 45 * it - sky
                    x = qbs[(tid, pid)]
                    x[0] += 1 if pid == lead[0] else 0
                    x[1] += att; x[2] += ay; x[3] += att + sk; x[4] = name
                    all_any[0] += ay; all_any[1] += att + sk
        lg = all_any[0] / all_any[1] if all_any[1] else 6.0
        for (tid, pid), (st, att, ay, dropbacks, name) in qbs.items():
            if st < 1 or dropbacks < 20:
                continue
            anya = ay / dropbacks
            v = max(1.5, min(cfg["qb_cap"], cfg["qb_base"] + cfg["qb_slope"] * (anya - lg)))
            vals[tid][pid] = (name, round(v, 2), round(st / max(tg[tid], 1), 3))
    names = {(tid, n.lower()): pid for tid, d in vals.items() for pid, (n, v, s) in d.items()}
    return vals, names


def injury_adjust(tid, inj, vals, names, cfg):
    """Points to subtract from a team for listed injuries; returns (points, [text])."""
    tot, det = 0.0, []
    for aid, name, w, status in inj.get(tid, []):
        pid = aid if aid in vals.get(tid, {}) else names.get((tid, name.lower()))
        if not pid or pid not in vals.get(tid, {}):
            continue
        nm, v, share = vals[tid][pid]
        pts = v * w * share
        if pts >= 0.2:
            tot += pts
            det.append(f"{nm} {status} -{pts:.1f}")
    cap = cfg.get("team_inj_cap") or cfg.get("qb_cap") or 10
    return min(tot, cap), det


def last_lock(cfg, now):
    day, h, m = cfg["lock"]
    base = now.replace(hour=h, minute=m, second=0, microsecond=0)
    for k in range(8):
        d = base - timedelta(days=k)
        if d <= now and (day == "daily" or DAYS[d.weekday()] == day):
            return d


def next_lock(cfg, now):
    day, h, m = cfg["lock"]
    base = now.replace(hour=h, minute=m, second=0, microsecond=0)
    for k in range(9):
        d = base + timedelta(days=k)
        if d > now and (day == "daily" or DAYS[d.weekday()] == day):
            return d


# ------------------------------- hockey goal model --------------------------
def pois(l, k):
    return math.exp(-l) * l ** k / math.factorial(k)


def hockey_dist(lh, la, maxg=12):
    """Final-score margin and total distributions (home perspective), regulation Poisson + overtime:
    a regulation tie becomes a 1-goal win for either side (50/50) and adds one goal to the total."""
    lh, la = max(lh, 0.3), max(la, 0.3)
    ph = [pois(lh, k) for k in range(maxg + 1)]
    pa = [pois(la, k) for k in range(maxg + 1)]
    mg, tg = defaultdict(float), defaultdict(float)
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            p = ph[i] * pa[j]
            if i == j:
                mg[1] += p / 2; mg[-1] += p / 2; tg[i + j + 1] += p
            else:
                mg[i - j] += p; tg[i + j] += p
    return mg, tg


def judge_hockey(g, m, tot, cfg, which):
    picks = []
    od = g.get("odds") or {}
    mg, tg = hockey_dist((tot + m) / 2, (tot - m) / 2)
    edge = cfg["ev_edge"]
    def ev(pw, pp, price):
        d = dec_from_american(price) or 1.909
        return pw * (d - 1) - (1 - pw - pp)
    def add(mkt, cands):
        best = max(cands, key=lambda c: c["edge"], default=None)
        if best and edge <= best["edge"] <= edge * MAX_GAP_X * 2:
            picks.append(best)
    sp = (od.get("spread") or {}).get(which)
    if sp:
        L, ho, ao = sp
        c = []
        for side, line, price in (("home", L, ho), ("away", -L, ao)):
            sgn_ = 1 if side == "home" else -1
            pw = sum(p for d, p in mg.items() if sgn_ * d + line > 0)
            pp = sum(p for d, p in mg.items() if sgn_ * d + line == 0)
            c.append(dict(mkt="spread", side=side, line=line, price=price, edge=round(ev(pw, pp, price), 4), model=round(pw, 4)))
        add("spread", c)
    tt = (od.get("total") or {}).get(which)
    if tt:
        T, oo, uo = tt
        po = sum(p for k, p in tg.items() if k > T); pu = sum(p for k, p in tg.items() if k < T); pp = 1 - po - pu
        add("total", [dict(mkt="total", side="over", line=T, price=oo, edge=round(ev(po, pp, oo), 4), model=round(tot, 2)),
                      dict(mkt="total", side="under", line=T, price=uo, edge=round(ev(pu, pp, uo), 4), model=round(tot, 2))])
    ml = (od.get("ml") or {}).get(which)
    if ml:
        phw = sum(p for d, p in mg.items() if d > 0)
        c = [dict(mkt="ml", side=s_, line=None, price=pr, edge=round(ev(pw, 0, pr), 4), model=round(pw, 4))
             for s_, pw, pr in (("home", phw, ml[0]), ("away", 1 - phw, ml[1])) if pw >= ML_MIN_P]
        add("ml", c)
    return picks


def win_prob(cfg, m, tot, sd):
    if cfg["kind"] == "hockey":
        mg, _ = hockey_dist((tot + m) / 2, (tot - m) / 2)
        return sum(p for d, p in mg.items() if d > 0)
    return N.cdf(m / sd)


# ------------------------------- value ---------------------------------------
def judge(g, m, tot, p_home, cfg, which="cur"):
    """Value picks against the given line set ('cur' = current DraftKings line). Returns list of picks."""
    if cfg["kind"] == "hockey":
        return judge_hockey(g, m, tot, cfg, which)
    picks = []
    od = g.get("odds") or {}
    thr_s, thr_t = cfg["spread_edge"], cfg["total_edge"]
    sp = (od.get("spread") or {}).get(which)
    if sp:
        L, ho, ao = sp
        e = m + L                                  # + : we like home more than the line does
        if thr_s <= abs(e) <= thr_s * MAX_GAP_X:
            side = "home" if e > 0 else "away"
            picks.append(dict(mkt="spread", side=side, line=L if side == "home" else -L, price=ho if side == "home" else ao,
                              edge=round(abs(e), 2), model=round(-m if side == "home" else m, 1)))
    tt = (od.get("total") or {}).get(which)
    if tt:
        T, oo, uo = tt
        e = tot - T
        if thr_t <= abs(e) <= thr_t * MAX_GAP_X:
            side = "over" if e > 0 else "under"
            picks.append(dict(mkt="total", side=side, line=T, price=oo if side == "over" else uo, edge=round(abs(e), 2),
                              model=round(tot, 1)))
    ml = (od.get("ml") or {}).get(which)
    if ml:
        best = None
        for side, p, price in (("home", p_home, ml[0]), ("away", 1 - p_home, ml[1])):
            d = dec_from_american(price)
            if not d or p < ML_MIN_P:
                continue
            ev = p * d - 1
            if ML_EDGE <= ev <= ML_EDGE * MAX_GAP_X * 2 and (best is None or ev > best["edge"]):
                best = dict(mkt="ml", side=side, line=None, price=price, edge=round(ev, 4), model=round(p, 4))
        if best:
            picks.append(best)
    return picks


def grade(mkt, side, line, hs, as_):
    if mkt == "total":
        t = hs + as_
        r = (t - line) if side == "over" else (line - t)
    else:
        mine = (hs - as_) if side == "home" else (as_ - hs)
        r = mine + (line or 0.0) if mkt == "spread" else mine
    return "W" if r > 0 else ("L" if r < 0 else "P")


def pl_of(res, price):
    d = dec_from_american(price) or 1.909
    return round(d - 1, 4) if res == "W" else (-1.0 if res == "L" else 0.0)


# ------------------------------- backtest -----------------------------------
def backtest(sk, cfg, c, feats, now):
    """Every finished game this season: ratings from earlier days only, judged at DraftKings' opening lines
    with the same value rules (no injuries/weather - those aren't available historically)."""
    days = defaultdict(list)
    for g in c["games"].values():
        if g["status"] == "final" and g["hs"] is not None and ((c["box"].get(g["id"]) or {}).get("dk")):
            days[parse_t(g["commence"]).date()].append(g)
    rows = []
    for d in sorted(days):
        cut = datetime(d.year, d.month, d.day, tzinfo=ET)
        F = fit(c, cfg, feats, cut, before=cut)
        for g in days[d]:
            if not (rated(F, g["home"]["id"]) and rated(F, g["away"]["id"])):
                continue
            em = nfelo_margin(g)[0] if sk == "nfl" else None
            m, tot, _ = predict(g, F, cfg, feats, elo_margin=em)
            rows.append((g, m, tot))
    sd = cfg["sd"]
    if len(rows) >= MIN_BT_GAMES and cfg["kind"] != "hockey":
        best = None
        for k in range(60, 151, 3):
            s = cfg["sd"] * k / 100
            ll = sum(logloss(N.cdf(m / s), 1 if g["hs"] > g["as_"] else 0) for g, m, t in rows if g["hs"] != g["as_"]) / len(rows)
            if best is None or ll < best[0]:
                best = (ll, s)
        sd = best[1]
    res = {mk: dict(n=0, w=0, l=0, p=0, units=0.0) for mk in ("spread", "total", "ml")}
    err = dict(sm=[0.0, 0.0, 0], tm=[0.0, 0.0, 0])
    for g, m, tot in rows:
        gg = dict(g, odds=(c["box"][g["id"]] or {}).get("dk"))
        sp = (gg["odds"] or {}).get("spread", {}).get("open")
        if sp:  # how close our line and the opening line were to the real margin
            err["sm"][0] += abs((g["hs"] - g["as_"]) - m); err["sm"][1] += abs((g["hs"] - g["as_"]) + sp[0]); err["sm"][2] += 1
        tt = (gg["odds"] or {}).get("total", {}).get("open")
        if tt:
            err["tm"][0] += abs((g["hs"] + g["as_"]) - tot); err["tm"][1] += abs((g["hs"] + g["as_"]) - tt[0]); err["tm"][2] += 1
        for pk in judge(gg, m, tot, win_prob(cfg, m, tot, sd), cfg, which="open"):
            r = grade(pk["mkt"], pk["side"], pk["line"], g["hs"], g["as_"])
            x = res[pk["mkt"]]
            x["n"] += 1; x[r.lower()] += 1; x["units"] += pl_of(r, pk["price"])
    for x in res.values():
        x["units"] = round(x["units"], 2)
        x["roi"] = round(x["units"] / (x["w"] + x["l"]), 4) if (x["w"] + x["l"]) else None
    mae = lambda e: (round(e[0] / e[2], 2), round(e[1] / e[2], 2)) if e[2] else (None, None)
    return dict(date=now.date().isoformat(), games=len(rows), sd=round(sd, 2), markets=res,
                spread_mae=mae(err["sm"]), total_mae=mae(err["tm"]))


# ------------------------------- ledger / store -----------------------------
def load_store():
    st = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            st = json.load(f)
    for k, v in (("bets", {}), ("locks", {}), ("models", {}), ("live", {}), ("wx", {})):
        st.setdefault(k, v)
    if st.get("track_v") != TRACK_VERSION:          # old opening-line tracking -> start the record fresh
        st["bets"], st["locks"] = {}, {}
        st.pop("judged", None)
        st["track_v"] = TRACK_VERSION
    st["notes"] = []
    return st


def save_store(st):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: st[k] for k in ("bets", "locks", "models", "live", "wx", "track_v")}, f, separators=(",", ":"))
    os.replace(tmp, DATA_FILE)


def lock_due(st, sk, cfg, now):
    """(lock_key, lock_until) when this run should lock tracked bets, else None.
    Regular locks: first run within LOCK_GRACE_MIN after the lock time; games until the next lock time.
    Weekly sports that have never locked lock once immediately (catch-up for the current week)."""
    lk = last_lock(cfg, now)
    if lk and st["locks"].get(sk) != lk.isoformat() and timedelta(0) <= now - lk <= timedelta(minutes=LOCK_GRACE_MIN):
        return lk.isoformat(), next_lock(cfg, lk + timedelta(minutes=1))
    if cfg["weekly"] and sk not in st["locks"]:
        return (lk or now).isoformat(), next_lock(cfg, now)
    return None


def cached_weather(st, g, geo, now):
    """Kickoff forecast, re-checked at most every 3 hours per game."""
    w = st["wx"].get(g["id"])
    if w and (now - parse_t(w["at"])) < timedelta(hours=3):
        return w["v"]
    v = weather(g, geo)
    st["wx"][g["id"]] = dict(at=now.isoformat(), v=v)
    return v


def label_of(g, side):
    return g[side]["abbr"] or g[side]["short"] or g[side]["location"]


def update(st, now):
    geo = load_geo()
    for sk, cfg in SPORTS.items():
        c = load_cache(sk)
        refresh_games(sk, cfg, c, now, st["notes"])
        refresh_boxes(sk, cfg, c)
        build_prior(sk, cfg, c, st["notes"])
        missing = {place(g) for g in c["games"].values() if g.get("city") and place(g) not in geo}
        if missing:
            geocode_all(geo, missing)
            save_geo(geo)
        feats = schedule_features(c, cfg, geo)
        if (c.get("bt") or {}).get("date") != now.date().isoformat():
            c["bt"] = backtest(sk, cfg, c, feats, now)
        sd = c["bt"]["sd"] if c["bt"] and c["bt"]["games"] >= MIN_BT_GAMES else cfg["sd"]
        # every upcoming game on the board is re-priced each run; at lock time, the lock's games too
        wa, wb = window(cfg, now)
        due = lock_due(st, sk, cfg, now)
        in_lock = lambda g: due is not None and now < parse_t(g["commence"]) < due[1]
        pre = [g for g in c["games"].values() if g["status"] == "pre" and parse_t(g["commence"]) > now and g.get("odds")
               and (wa <= parse_t(g["commence"]) < wb or in_lock(g))]
        locked_any = False
        if pre:
            F = fit(c, cfg, feats, now)
            inj_model = cfg["kind"] in ("bball", "fball") and sk != "nfl"   # NFL QBs via nfelo; NHL: no injury model
            vals, names = player_values(c, cfg) if inj_model else ({}, {})
            sums = pmap(lambda gid: FETCH(SUMMARY.format(cfg["path"], gid)), [g["id"] for g in pre]) if inj_model else {}
            for g in pre:
                if not (rated(F, g["home"]["id"]) and rated(F, g["away"]["id"])):
                    continue
                inj = parse_injuries(sums.get(g["id"]) or {}) if sums.get(g["id"]) else {}
                ih, dh = injury_adjust(g["home"]["id"], inj, vals, names, cfg) if vals else (0.0, [])
                ia, da = injury_adjust(g["away"]["id"], inj, vals, names, cfg) if vals else (0.0, [])
                em, qbs = nfelo_margin(g) if sk == "nfl" else (None, None)
                wx = cached_weather(st, g, geo, now) if cfg["kind"] == "fball" else None
                m, tot, parts = predict(g, F, cfg, feats, (ih, ia), wx, em)
                p_home = win_prob(cfg, m, tot, sd)
                why = feats.get(g["id"], (0.0, []))[1] + dh + da + ([qbs] if qbs else []) + ([parts["wx"]] if parts["wx"] else [])
                st["models"][g["id"]] = dict(sport=sk, at=now.isoformat(), spread=round(-m, 1), total=round(tot, 1),
                                             p_home=round(p_home, 4), parts=parts, why=why)
                # board highlights: value vs the CURRENT line, refreshed every run until kickoff (then frozen)
                st["live"][g["id"]] = dict(at=now.isoformat(), picks=[{k: p[k] for k in ("mkt", "side", "line", "price")}
                                                                      for p in judge(g, m, tot, p_home, cfg, which="cur")])
                # tracked bets: locked at the sport's lock time, at the CURRENT line, once per game/market
                if in_lock(g):
                    locked_any = True
                    for p in judge(g, m, tot, p_home, cfg, which="cur"):
                        key = f"{g['id']}|{p['mkt']}"
                        if key in st["bets"]:
                            continue
                        team = (label_of(g, p["side"]) if p["side"] in ("home", "away") else p["side"].capitalize())
                        st["bets"][key] = dict(id=key, sport=sk, event_id=g["id"], mkt=p["mkt"], side=p["side"], team=team,
                                               line=p["line"], price=p["price"], edge=p["edge"], model=p["model"],
                                               ev=cfg["kind"] == "hockey" or p["mkt"] == "ml",
                                               game=f"{label_of(g,'away')} {'vs' if g['neutral'] else '@'} {label_of(g,'home')}",
                                               commence=g["commence"], locked_at=now.isoformat(), lock=due[0], units=1.0,
                                               result=None, pl=None, final=None, graded_at=None)
        if due is not None:
            n_lock = sum(1 for g in c["games"].values() if g["status"] == "pre" and in_lock(g))
            if locked_any or n_lock == 0:
                st["locks"][sk] = due[0]
            else:
                st["notes"].append(f"{cfg['label']}: no lines/ratings at lock time - will retry next run.")
        # grade once, never again
        for b in [b for b in st["bets"].values() if b["sport"] == sk and b["result"] is None]:
            g = c["games"].get(b["event_id"])
            if not g or (now - parse_t(b["commence"])).total_seconds() < FINAL_AFTER_H * 3600:
                continue
            if g["status"] == "final" and g["hs"] is not None:
                r = grade(b["mkt"], b["side"], b["line"], g["hs"], g["as_"])
                b.update(result=r, pl=pl_of(r, b["price"]), final=f"{g['as_']:g}-{g['hs']:g}", graded_at=now.isoformat())
            elif g["status"] == "off":
                b.update(result="P", pl=0.0, final="postponed", graded_at=now.isoformat())
        save_cache(sk, c)


# ------------------------------- output -------------------------------------
def build_page(st, now):
    sports = {}
    for sk, cfg in SPORTS.items():
        c = load_cache(sk)
        a, b = window(cfg, now)
        games = []
        for g in sorted(c["games"].values(), key=lambda g: (g["commence"], g["id"])):
            t = parse_t(g["commence"])
            if not (a <= t < b):
                continue
            lv = []
            for p in (st["live"].get(g["id"]) or {}).get("picks", []):
                r = grade(p["mkt"], p["side"], p["line"], g["hs"], g["as_"]) if g["status"] == "final" and g["hs"] is not None else None
                lv.append(dict(mkt=p["mkt"], side=p["side"], result=r))
            games.append(dict(id=g["id"], commence=g["commence"], status=g["status"], detail=g.get("detail", ""),
                              neutral=g["neutral"], venue=g.get("venue", ""), home=g["home"], away=g["away"],
                              hs=g["hs"], as_=g["as_"], odds=g.get("odds"), model=st["models"].get(g["id"]), value=lv))
        d_, h_, m_ = cfg["lock"]
        lock_txt = ("Daily" if d_ == "daily" else d_.capitalize()) + " " + datetime(2000, 1, 1, h_, m_).strftime("%-I%p").lower() + " ET"
        sports[sk] = dict(label=cfg["label"], kind=cfg["kind"], weekly=cfg["weekly"], window=[a.isoformat(), b.isoformat()], games=games,
                          lock=lock_txt, next_lock=next_lock(cfg, now).isoformat(),
                          bt=c.get("bt"), edges=dict(spread=cfg.get("spread_edge"), total=cfg.get("total_edge"), ml=ML_EDGE, ev=cfg.get("ev_edge")))
    bets = sorted(st["bets"].values(), key=lambda x: (x["commence"], x["id"]))
    payload = dict(generated=now.isoformat(), sports=sports, bets=bets, notes=sorted(set(st["notes"]))[:8])
    with open(TEMPLATE) as f:
        html = f.read().replace("/*__DATA__*/null", json.dumps(payload))
    with open(OUT_HTML, "w") as f:
        f.write(html)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    now = datetime.now(ET)
    if "--now" in args:
        now = datetime.fromisoformat(args[args.index("--now") + 1]).replace(tzinfo=ET)
    st = load_store()
    if "--offline" not in args:
        update(st, now)
        save_store(st)
    build_page(st, now)
    g = [b for b in st["bets"].values() if b["result"]]
    print(f"{now:%a %b %-d %-I:%M %p} ET | {len(st['bets'])} bets | {sum(b['pl'] for b in g):+.2f}u on {len(g)} graded")
    for n in st["notes"]:
        print("  note:", n)
    if "--no-open" not in args:
        webbrowser.open("file://" + OUT_HTML)


if __name__ == "__main__":
    main()
