# Git_CICD_Demo

A small Python project used for MLOps Experiment 2: Git, GitHub and CI/CD. It shows how version control and an automated test workflow work together to keep code changes reliable.

## What it does

`app.py` contains `predict_label(value)`, a tiny stand-in for a model prediction step. It returns `"positive"` when the value is 0.5 or higher, and `"negative"` otherwise. `test_app.py` checks both cases with pytest.

## Project structure

```
Git_CICD_Demo/
├── app.py                 # Prediction function and demo entry point
├── test_app.py            # pytest unit tests
├── requirements-dev.txt   # Development dependencies (pytest)
└── README.md              # This file
```

The CI workflow is stored in the repository root at `.github/workflows/python-ci.yml`, because GitHub only reads workflows from there.

## Setup

Requires Python 3.9 or later (the CI workflow uses 3.11).

```bash
cd Git_CICD_Demo
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Run the app

```bash
python3 app.py
```

Expected output:

```
Prediction: positive
```

## Run the tests

```bash
python3 -m pytest -v
```

Expected result: 2 tests passed.

## Continuous Integration

The **Python CI** workflow (GitHub Actions) runs automatically:

- on every push to `main`
- on every pull request targeting `main`

Each run checks out the code, sets up Python 3.11, installs `requirements-dev.txt` and runs `pytest -v`. A failing test turns the check red and the change should not be merged until it is fixed.

## Git workflow used

1. Create a feature branch: `git switch -c feature-message`
2. Edit the code, then `git add app.py` and `git commit -m "Update prediction demo"`
3. Push the branch: `git push -u origin feature-message`
4. Open a pull request into `main` and wait for the CI check
5. Fix any failure with a new commit on the same branch
6. Merge once the check is green, then `git switch main` and `git pull origin main`

## Notes

- This folder is a subdirectory of the larger `MLOps` repository, so the remote is named `MLOps`.
- Do not commit passwords or access tokens. Use your Git credential manager or GitHub secrets.