# Cinema Show Occupancy Prediction - MP183

![Status](https://img.shields.io/badge/status-in%20progress-orange)
![Problem](https://img.shields.io/badge/problem-MP183-blue)

An MLOps mini-project for predicting the occupancy of cinema shows in Jaipur. The
project covers data quality analysis, feature engineering, model training and
tracking, drift monitoring, retraining, API deployment, containerization,
Kubernetes deployment, and final holdout predictions.

## Project Information

| Field | Value |
|---|---|
| Group | G010 |
| Problem | MP183 - Cinema Show Occupancy (Jaipur) |
| Faculty | Amit Soni |
| Batch | 31 |
| Target | `occupancy_pct` |
| Evaluation metric | Mean Absolute Error (MAE) |

### Team

| Student ID | GitHub Username |
|---|---|
| S24CSEU0950 | vrishankpathak |
| S24CSEU0806 | DivjyotNandaSingh |
| S24CSEU0917 | harshkumar |
| S24CSEU0750 | HardikChawla |

## Business Problem

Cinema owners in Jaipur choose show timings one week ahead. The goal is to
predict the percentage of seats filled for each show so that owners can reduce
shows for weak films and add shows for popular films.

This is a supervised regression problem. Predictions are measured using MAE,
where lower values indicate better estimates.

## Dataset

| File | Period | Purpose |
|---|---|---|
| `data/train.csv` | January-June 2026 | Messy historical training data |
| `data/later.csv` | July-August 2026 | Post-launch monitoring and retraining data |
| `data/holdout_features.csv` | September 2026 | Final prediction input without the target |

For DVC, raw copies of the training and later datasets are tracked at
`data/raw/train.csv` and `data/raw/later.csv`. The original `data/` paths are
retained for compatibility with the problem specification.

The available feature columns are:

- `record_id`
- `date`
- `ticket_price_rs`
- `days_since_release`
- `star_rating`
- `show_hour`
- `screen_seats`
- `distance_to_metro_km`
- `language`
- `location`

`tickets_scanned_pct` is present in the historical data but must not be used as
a model feature because it is known only after the show is complete.
`occupancy_pct` is the regression target.

## Intended Project Structure

```text
.
├── data/
│   ├── train.csv
│   ├── later.csv
│   ├── holdout_features.csv
│   └── raw/
│       ├── train.csv.dvc
│       └── later.csv.dvc
├── app/
│   └── main.py
├── src/
│   ├── clean.py
│   ├── findings.py
│   ├── train.py
│   ├── monitor.py
│   └── predict.py
├── tests/
│   └── test_main.py
├── predictions/
├── k8s/
│   └── deployment.yaml
├── Dockerfile
├── requirements.txt
├── params.yaml
├── reports/
│   └── findings.yaml
├── PROBLEM.md
├── GROUP.md
└── README.md
```

The structure above describes the expected implementation. Paths and filenames
may be adjusted while preserving the required outputs.

## Data Quality Pipeline

The training data must be inspected for these seven issue types:

1. Target leakage
2. Unit mix-up
3. Sentinel values
4. Duplicate rows
5. Stuck sensors
6. Target typos
7. Spelling variants

All evidence and detected values must be computed by code in `src/` and recorded
in `reports/findings.yaml`. The final submission should not rely on manually
guessed results.

## Training and Experiment Tracking

The v1 model is trained only on cleaned `data/train.csv`.

The project uses MLflow to:

- log preprocessing details, model parameters, and MAE;
- save the v1 experiment run;
- register the selected model;
- promote v2 to the `@champion` alias only when it meets the promotion rule.

Before running commands, configure the required Python environment and MLflow
experiment settings for the workspace.

Example workflow:

```powershell
python -m pip install -r requirements.txt
pytest
python src/clean.py
python src/train.py
python src/monitor.py
python src/predict.py
```

Actual script names may differ based on the implementation.

## Monitoring and Retraining

After the v1 model goes live:

1. Calculate Population Stability Index (PSI) for every numeric feature between
   cleaned `train.csv` and `later.csv`, using 10 bins.
2. Calculate v1 MAE separately for `City Centre`, `Mall`, and `Highway`.
3. Sort `later.csv` chronologically.
4. Train v2 on cleaned `train.csv` plus the first 70% of `later.csv`.
5. Evaluate v1 and v2 on the remaining 30%.
6. Promote v2 only when:

```text
MAE(v2) <= 0.88 * MAE(v1)
```

The monitoring decisions and metric values must be written to
`reports/findings.yaml`.

## Prediction API

The prediction service should expose:

### `GET /health`

Returns:

```json
{"status": "ok"}
```

### `POST /predict`

Request fields:

```json
{
  "ticket_price_rs": 350,
  "days_since_release": 20,
  "star_rating": 4.2,
  "show_hour": 18,
  "screen_seats": 220,
  "distance_to_metro_km": 2.4,
  "language": "hindi",
  "location": "Mall"
}
```

The request schema must reject:

- `days_since_release > 70`;
- `screen_seats < 40`.

These constraints must be enforced with Pydantic `Field` validation, resulting
in HTTP 422 for invalid requests.

The API loads its serialized model from `models/champion.joblib` by default.
Set the `MODEL_PATH` environment variable to use a different model artifact.
Until a model artifact is available, `/predict` returns HTTP 503; `/health`
remains available for service health checks.

Example local API start command:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Replace the command if the implementation uses a different application module.

## Docker

The container image must use exactly:

```text
mp183-api:1.0
```

Example build and run commands:

```powershell
docker build -t mp183-api:1.0 .
docker run --rm -p 8000:8000 mp183-api:1.0
```

The container should listen on port `8000`.

## Kubernetes

The Kubernetes resources must use:

| Resource | Required value |
|---|---|
| Deployment | `mp183-api` |
| Service | `mp183-svc` |
| Replicas | `3` |
| Container port | `8000` |
| Service port | `80` |
| Readiness probe | `GET /health` |
| Liveness probe | `GET /health` |

The Service forwards port `80` to container port `8000`. The liveness probe
waits longer than the readiness probe.

Example deployment command:

```powershell
kubectl apply -f k8s/
kubectl get deployment mp183-api
kubectl get pods
kubectl get service mp183-svc
```

## Final Predictions

Generate predictions for every row in `data/holdout_features.csv` using the
MLflow `@champion` model.

Save the result as:

```text
predictions/holdout_predictions.csv
```

The file must contain exactly these columns:

```csv
record_id,prediction
```

The predictions must correspond to one row per `record_id` from the holdout
file.

## Submission Checklist

- [ ] Data issue detection code is complete
- [ ] `reports/findings.yaml` contains verified values
- [ ] v1 is trained and registered in MLflow
- [ ] PSI and location-wise MAE are recorded
- [ ] v2 retraining and promotion decision are complete
- [ ] `/health` and `/predict` are implemented
- [ ] Pydantic validation rejects invalid requests
- [ ] Docker image is tagged `mp183-api:1.0`
- [ ] Kubernetes Deployment and Service use the required names and ports
- [ ] Three replicas are configured
- [ ] Readiness and liveness probes are configured
- [ ] `predictions/holdout_predictions.csv` has the exact required format

## Reference

- [Problem Specification](PROBLEM.md)
- [Group Details](GROUP.md)
- [Findings Template](reports/findings.yaml)
