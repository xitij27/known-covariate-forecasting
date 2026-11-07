# Understanding the problem

Before building any model I spent time just looking at the data. I'm glad I did — most of what I figured out later was already visible if I looked closely enough.

## The shape of the data

- **96 series.** Each one is a separate "unit" with its own characteristics. They all share the same time period.
- **180 days × 24 hours = 4,320 rows per series.** All series cover the same dates.
- **A `target` column** — the thing I need to predict. Values range from roughly 0 to 50, with per-series means somewhere between 6 and 14.
- **11 forecast columns**: `queue_pressure_forecast`, `network_pressure_forecast`, `workload_intensity`, `demand_forecast`, `shock_risk`, `maintenance_known`, `unit_reliability_forecast`, `event_load_forecast`, `service_irregularity_risk_forecast`, `upstream_quality_forecast`, `promotion_intensity`. These are pre-computed estimates of various conditions.
- **A few "static" columns**: `nominal_capacity` (one fixed value per series), `zone_sin/cos` (some location encoding), `trend` (a slow linear drift over time).
- **Time encodings** that come pre-computed: hour-of-day, day-of-week, weekend flag.

The validation file (`validation_input.csv`) has the same forecast columns and time columns for the 14-day forecast window — **but no `target` column.** I had to produce it.

## What I looked at first

I computed the correlation of each forecast feature with the target, both pooled across all series and within each series. The numbers were striking:

| feature | r² with target |
|---|---|
| `queue_pressure_forecast` | 45% |
| `network_pressure_forecast` | 43% |
| `workload_intensity` | 29% |
| `shock_risk` | 25% |
| `event_load_forecast` | 17% |

Just two features explained nearly 90% of the target's variance between them. That is a lot.

A few features were nearly useless:

| feature | correlation with target |
|---|---|
| `staffing_forecast` | r² = 0.2% |
| `throughput_disruption_risk_forecast` | r² = 0.1% |
| `nominal_capacity` | r ≈ 0 |
| `trend` | r ≈ +0.05 |

So `nominal_capacity` is essentially noise — its variation doesn't relate to the target's variation across units. And the target has no real trend over the 180 days; when I chopped the period into thirds and computed the mean of each chunk they came out 8.93, 9.40, 9.09. Basically flat.

## Things I noted but didn't appreciate yet

- **The forecast columns had small NaN gaps.** About 4–5% of values in each column. The gaps were short (average 1.1 hours, max 3 hours) and seemed independent across columns. So if I dropped rows with any NaN, I'd lose a lot of data — about 40% of training rows.
- **The target is stationary.** Chunk means barely move over 180 days. This will matter later when I worry about whether the validation period drifts away from training.
- **Distribution shift between the training tail and the val period was tiny** — under 0.25 standard deviations on every feature. Whatever I trained on the last 14 days of train should generalize to the val window.
- **The "forecast" columns have informative names.** `queue_pressure_forecast` sounds like *someone already forecast queue pressure for you*. I didn't ask the right question yet: "If I have a forecast of queue pressure for the val period, what is left for me to forecast?"

## How this shaped my next move

At this point I had two pieces of information that I now think I should have weighted more heavily:

1. The forecast features explain most of the target. Loudly.
2. The validation input has those features, but no target.

Instead, what I did next was: build an LSTM. Because I was thinking "time series → sequence model." I'll get to why that was the wrong instinct.
