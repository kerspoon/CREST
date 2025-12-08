# Future Ideas

Ideas for extending the CREST demand model.

---

## EV Charging Module

Simple placeholder model for electric vehicle charging.

**Parameters:**
- Power: 7 kW (fixed)
- Duration: X ~ Normal(3h, 2h)
- Start time: Y ~ Normal(6pm, 1h)
- Penetration: 10% of households

**Implementation:** ~100 lines in new `ev.py`, ~15 lines integration in `dwelling.py`, add `has_ev` column to Dwellings.csv.

---

## Project Ideas

### BEng Level

| Project | Description |
|---------|-------------|
| Battery storage | Add home battery with control strategies (self-consumption, TOU tariff, peak shaving) |
| Heat pump retrofit | Replace gas boiler with ASHP, COP varying with outdoor temp |
| Demand response | Identify shiftable loads, quantify flexibility potential |

### Masters Level

| Project | Description |
|---------|-------------|
| ML surrogate model | Train neural net on 100k runs for instant predictions |
| Neighbourhood aggregation | Simulate 1000+ homes, diversity factor analysis |
| Climate change scenarios | 2050/2080 weather projections, heating→cooling shift |
| Multi-country | Adapt occupancy/appliances for different countries |
| Real-time digital twin | Calibrate to smart meter data, predictive control |

---

## Quick Wins

| Improvement | Effort |
|-------------|--------|
| Parallel Monte Carlo (multiprocessing) | 1 day |
| Web API (FastAPI) | 1 week |
| Interactive dashboard (Streamlit) | 1 week |
| pip-installable package | 2 days |
| Unit test coverage | 1 week |
