# gta6-trophy-forecast-sakura

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hellolucyheart/gta6-trophy-forecast-sakura/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hellolucyheart/gta6-trophy-forecast-sakura/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate one week out, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. An earn rate is earners divided by players tracked, and the two halves move for different reasons in launch week, so the program models them separately in log space — log earners and log players tracked each fitted against log time by recency-weighted least squares — and blends that structural forecast with a direct trend on the logit rate. The median's move from the last read is shrunk and capped, saturated trophies at 100% get a dilution prior as the tracked base grows, and the spread combines a random walk, fit residuals, trend uncertainty and the largest refresh step seen, so the band survives an Exophase refresh it has not seen. Python 3 standard library only, fully deterministic, ~0.1 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=68wvraae&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
