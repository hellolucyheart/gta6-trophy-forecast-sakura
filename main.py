"""Trophy-rarity forecaster for gta6-trophy-rarity-forecast-01.

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at a case's target instant, as five quantiles (5%, 25%,
50%, 75%, 95%). One run reads every <id>.in in the input directory and
writes one <id>.out per case. Python 3 standard library only, fully
deterministic, no network, no randomness.

How it works
------------
Exophase's earn rate for a trophy is (players who earned it) / (players
tracked). Both halves move during launch week, and they move for different
reasons, so the forecast models them separately and recombines them:

1.  Dilution and progress, in log space. For each read, with t the hours
    since the game's release (2026-11-19T00:00:00Z) and x = ln(1 + t):
      - the earners E = rate * N / 100, where N is the players tracked;
      - ln E and ln N are each fitted against x by recency-weighted least
        squares over the last 48 hours of reads (12-hour half-life), which
        makes the fit follow the latest refresh regime instead of launch
        night's stale page;
      - the structural forecast is rate = 100 * E(x_target) / N(x_target).
    A trophy whose earners grow slower than the tracked base dilutes
    downwards; one whose earners keep pace holds its rate. Using ln(1 + t)
    instead of t keeps the extrapolation concave: launch-week growth bends
    over instead of running away in a straight line.

2.  A direct trend on the logit rate. logit(rate) is fitted against the
    same x by the same weighted least squares, and read at the target.
    The median is the average of the structural and the direct forecast in
    logit space, so neither model can drag the answer alone. The move from
    the last read's rate is shrunk by 0.8 and capped at 2 logit units:
    a 4.5-day extrapolation of a 2.5-day history must not be allowed to
    sprint.

3.  Saturated trophies. A trophy sitting at 100% cannot rise, only dilute
    as new tracked players arrive who have not earned it yet. When the
    tracked base is forecast to grow, its median is pulled halfway toward
    logit(100 * (N_now / N_target) ** 0.15), a mild dilution prior: at
    100x growth that cap is 50%, at 10x it is 71%.

4.  Spread. The logit spread s combines, in quadrature: the baseline
    random walk 0.5 * sqrt(horizon in days); the residual scatter of the
    logit fit; a trend-uncertainty term that grows with the extrapolation
    distance in x; and half the largest single-read logit jump seen in the
    window (Exophase refreshes arrive as steps, and the answer is the
    first read in the target hour, so a refresh may still land between
    the last read and the target). s is floored at 0.55 and capped at 2.0
    logit units, so even a 90-minute practice case carries a band wide
    enough to survive a refresh it has not seen.

Fallbacks: with fewer than three usable reads, or no variation in x, the
median is the last read's rate (persistence) with the same spread rule.
A trophy missing from every read takes its tier's median rate from the
last read. A missing history takes tier priors (platinum 0.5%, gold 1%,
silver 1%, bronze 1%) with a wide spread.

Output: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
one row per trophy, quantiles non-decreasing, each 0..100, rounded to
4 decimals half-up. At most 65,536 bytes by construction.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is shipped; kept for parity).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z_BY_LEVEL = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

# Release instant: GTA VI launched 19 November 2026 (UTC day start).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Fit window and recency weighting.
WINDOW_HOURS = 48.0
HALF_LIFE_HOURS = 12.0
MIN_READS_FOR_FIT = 3
MIN_X_RANGE = 0.05

# Median-move guardrails, in logit units.
MOVE_SHRINK = 0.8
MAX_MOVE = 2.0

# Spread pieces, in logit units.
WALK_SIGMA = 0.5          # baseline random walk per sqrt(day)
SPREAD_FLOOR = 0.55
SPREAD_CAP = 2.0
TREND_UNC_PER_X = 0.35    # trend uncertainty per unit of extrapolation in x
JUMP_FRACTION = 0.5       # share of the largest observed step kept as spread
DEFAULT_RESID = 0.30      # residual scatter assumed when no fit exists
SATURATED_RATE = 99.0     # a trophy at/above this only dilutes
SATURATED_DILUTION = 0.15  # exponent of the dilution prior for saturated trophies

# Tier priors, used only when a trophy (or the whole history) has no rate.
TIER_PRIOR = {"platinum": 0.5, "gold": 1.0, "silver": 1.0, "bronze": 1.0}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment: datetime) -> float:
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def wls(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std) or None.

    Needs at least three points and real variation in x; with a degenerate
    design the caller falls back to persistence.
    """
    n = len(xs)
    if n < MIN_READS_FOR_FIT:
        return None
    if max(xs) - min(xs) < MIN_X_RANGE:
        return None
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(
        sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    )
    return a, b, resid


def tier_median_rate(trophies: list[dict], rates: list[float | None],
                     tier: str) -> float:
    same = [r for t, r in zip(trophies, rates)
            if t["tier"] == tier and r is not None]
    if same:
        same.sort()
        return same[len(same) // 2]
    known = sorted(r for r in rates if r is not None)
    if known:
        return known[len(known) // 2]
    return TIER_PRIOR.get(tier, 1.0)


def forecast_trophy(idx: int, trophy: dict, case: dict,
                    reads: list[dict], x_target: float,
                    last_rates: list[float | None]) -> dict:
    tier = trophy["tier"]
    last_rate = last_rates[idx]
    if last_rate is None:
        last_rate = tier_median_rate(case["trophies"], last_rates, tier)
    y_last = logit(last_rate)

    # Per-read series for this trophy: (x, logit rate, log earners, log N).
    xs, ys, ln_e, ln_n = [], [], [], []
    for read in reads:
        rate = read["rates"][idx]
        if rate is None:
            continue
        n_players = read.get("players_tracked")
        t = hours_since_release(parse_time(read["at"]))
        xs.append(math.log1p(t))
        ys.append(logit(rate))
        if isinstance(n_players, (int, float)) and n_players > 0:
            earners = max(rate, 0.005) * n_players / 100.0
            ln_e.append(math.log(max(earners, 0.5)))
            ln_n.append(math.log(n_players))
        else:
            ln_e.append(float("nan"))
            ln_n.append(float("nan"))

    # Recency weights over the window, newest read weight 1.
    t_last = hours_since_release(parse_time(reads[-1]["at"]))
    weights = [0.5 ** ((t_last - hours_since_release(parse_time(r["at"])))
                       / HALF_LIFE_HOURS) for r in reads]
    # Keep only reads that contributed a rate, aligned with xs/ys.
    w_kept = []
    for k, read in enumerate(reads):
        if read["rates"][idx] is not None:
            w_kept.append(weights[k])

    fit_y = wls(xs, ys, w_kept) if len(xs) >= MIN_READS_FOR_FIT else None
    ok_en = [i for i in range(len(xs)) if not math.isnan(ln_e[i])]
    fit_e = (wls([xs[i] for i in ok_en], [ln_e[i] for i in ok_en],
                 [w_kept[i] for i in ok_en]) if len(ok_en) >= MIN_READS_FOR_FIT
             else None)
    fit_n = (wls([xs[i] for i in ok_en], [ln_n[i] for i in ok_en],
                 [w_kept[i] for i in ok_en]) if len(ok_en) >= MIN_READS_FOR_FIT
             else None)

    x_last = xs[-1] if xs else math.log1p(t_last)
    dx = max(0.0, x_target - x_last)
    horizon_days = case["horizon_minutes"] / 1440.0

    # --- median ---------------------------------------------------------
    candidates = []
    if fit_y is not None:
        candidates.append(fit_y[0] + fit_y[1] * x_target)
    n_target = None
    if fit_e is not None and fit_n is not None:
        e_target = math.exp(fit_e[0] + fit_e[1] * x_target)
        n_target = math.exp(fit_n[0] + fit_n[1] * x_target)
        candidates.append(logit(100.0 * e_target / max(n_target, 1.0)))
    if candidates:
        y_med = sum(candidates) / len(candidates)
    else:
        y_med = y_last

    # Shrink and cap the move from the last read.
    move = (y_med - y_last) * MOVE_SHRINK
    move = max(-MAX_MOVE, min(MAX_MOVE, move))
    y_med = y_last + move

    # Saturated trophies only dilute as the tracked base grows.
    if last_rate >= SATURATED_RATE and n_target is not None:
        n_now = reads[-1].get("players_tracked")
        if isinstance(n_now, (int, float)) and n_now > 0 and n_target > n_now:
            cap = logit(100.0 * (n_now / n_target) ** SATURATED_DILUTION)
            if cap < y_med:
                y_med = 0.5 * (y_med + cap)

    # --- spread ---------------------------------------------------------
    walk = WALK_SIGMA * math.sqrt(horizon_days)
    resid = fit_y[2] if fit_y is not None else DEFAULT_RESID
    trend_unc = TREND_UNC_PER_X * dx
    jump = 0.0
    for a, b in zip(ys, ys[1:]):
        jump = max(jump, abs(b - a))
    jump *= JUMP_FRACTION
    if not ys:
        jump = DEFAULT_RESID
    spread = math.sqrt(walk ** 2 + resid ** 2 + trend_unc ** 2 + jump ** 2)
    spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread))

    z = [Z_BY_LEVEL.get(level, 0.0) for level in case["levels"]]
    qs = [round4(min(100.0, max(0.0, sigmoid(y_med + zi * spread))))
          for zi in z]
    # Rounding can only tie neighbours, never invert them; enforce it anyway.
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return {"id": trophy["id"], "q": qs}


def forecast_case(case: dict, name: str) -> dict:
    if not case["history"]:
        # No reads at all: tier priors with a wide spread.
        rows = []
        for trophy in case["trophies"]:
            prior = TIER_PRIOR.get(trophy["tier"], 1.0)
            y = logit(prior)
            z = [Z_BY_LEVEL.get(level, 0.0) for level in case["levels"]]
            qs = [round4(min(100.0, max(0.0, sigmoid(y + zi * 1.5))))
                  for zi in z]
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = math.log1p(max(0.0, (target - RELEASE).total_seconds() / 3600.0))

    # Window: the last 48 hours of reads; fewer than four, take them all.
    window_start = checkpoint.timestamp() - WINDOW_HOURS * 3600.0
    recent = [r for r in case["history"]
              if parse_time(r["at"]).timestamp() >= window_start]
    reads = recent if len(recent) >= 4 else list(case["history"])
    reads.sort(key=lambda r: r["at"])

    last_rates = list(reads[-1]["rates"])
    rows = [forecast_trophy(i, trophy, case, reads, x_target, last_rates)
            for i, trophy in enumerate(case["trophies"])]
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name} forecast exceeds 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
