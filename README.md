# OOP with Python 
MSCS · OOP with Python · Advent 2026

## Repository layout

```
.
├── README.md
├── requirements.txt
├── pytest.ini              # puts the repo root on the path so tests can `import src`
├── project1_population.ipynb
├── src/
│   ├── __init__.py
│   └── population.py
├── tests/
│   └── test_population.py
└── figures/                # PNGs written by the notebooks
```

## Summary of findings

### Mini-Project 1 — District Population Forecaster

* **Fastest relative growth: Wakiso** (CAGR 6.5 %/yr). Kampala and Gulu share an identical CAGR (4.6 %) because both grew by a factor of 1.5, which illustrates that CAGR ignores the path taken.
* **Model selection by hold-out (train 2015–21, test 2022–24):** exponential/CAGR growth won for Kampala, Wakiso, Gulu and Arua (test MAPE 0.4–4.1 %), and a linear trend won for Mbarara (MAPE 0.2 %). The **Fibonacci-ratio** model failed everywhere (MAPE 78–92 %) because it forces growth of about 62 % a year and learns nothing from the data.
* **Classrooms needed by 2029** (18 % school-age, 53 pupils per room): about **4,960 in total** (95 % range 4,280–5,640). The largest single need is in Wakiso (≈ 2,090), not Kampala (≈ 1,540), because growth rather than size drives new demand.
* A smooth forecast has low variance, but it is not low-*uncertainty*. The bootstrap intervals (1,000 residual resamples) show that Arua's forecast, distorted by an influx shock, carries an interval of about 18 % of the forecast value, against under 1 % for Mbarara.

## Data sources

All data are **illustrative** values taken from the assignment brief. The two extra districts in this project (Mbarara, Arua) are my own illustrative series, not UBOS figures.

## Declaration of AI use

> **Edit this section so that it describes accurately how you used AI.** The draft below is a starting point.

I used Claude (Anthropic) as a coding assistant for this assignment. It helped design the class structure, draft the module code, tests and notebook cells, and suggest wording for the analysis. I reviewed, ran and checked every result — including the hand calculations in the notebooks — and I can explain every line of the code.
