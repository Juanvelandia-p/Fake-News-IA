# Fake-News-IA

> **AI-powered Spanish-language fake news detection using classical ML and Transformer-based deep learning.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikitlearn)](https://scikit-learn.org/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Value Proposition](#value-proposition)
3. [Use Cases & Intended Audience](#use-cases--intended-audience)
4. [Main Features](#main-features)
5. [Architecture](#architecture)
6. [Non-Functional Requirements](#non-functional-requirements)
7. [Project Structure](#project-structure)
8. [Setup & Installation](#setup--installation)
9. [Environment Variables & Safe Configuration](#environment-variables--safe-configuration)
10. [Running the Pipeline](#running-the-pipeline)
11. [Data](#data)
12. [Testing & Coverage](#testing--coverage)
13. [Deployment](#deployment)
14. [CI/CD](#cicd)
15. [Contribution Guidelines](#contribution-guidelines)
16. [License](#license)

---

## Project Overview

**Fake-News-IA** is an end-to-end machine learning pipeline for detecting fake news written in Spanish. It combines:

- A **classical baseline** — a TF-IDF + LinearSVC pipeline for fast, interpretable classification.
- An **advanced Transformer model** — fine-tuned [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) (Spanish BERT), the state-of-the-art pre-trained language model for Spanish, for high-accuracy sequence classification.

The system ingests heterogeneous datasets (Excel and CSV), performs NLP-grade text preprocessing in Spanish, trains both models under class-imbalance correction, and produces rigorous evaluation metrics including AUC-ROC and optimal F1-threshold analysis.

---

## Value Proposition

| Problem | This Solution |
|---|---|
| Spanish fake news is under-served by existing English-focused tools | Built entirely for Spanish-language corpora |
| Simple keyword filters have high false-positive rates | Deep contextual understanding via BETO Transformer |
| Class imbalance distorts model fairness | Computed class weights applied during BERT training; balanced sampling for SVM |
| Reproducing ML results requires careful environment management | Fully reproducible pipeline with pinned dependencies |

---

## Use Cases & Intended Audience

**Primary use cases:**
- Automated pre-screening of news articles for editorial fact-checking teams.
- Academic research on Spanish NLP, disinformation detection, and bias mitigation.
- Integration as a back-end classification service in content moderation platforms.
- Benchmark comparisons between classical ML and Transformer approaches on imbalanced datasets.

**Intended audience:**
- Data scientists and ML engineers working on NLP classification tasks.
- Researchers in computational journalism and media literacy.
- Backend engineers integrating AI-driven content moderation into production systems.
- Technical recruiters evaluating applied AI / NLP competencies.

---

## Main Features

| Feature | Details |
|---|---|
| **Spanish NLP preprocessing** | Lowercasing, URL removal, punctuation stripping, stop-word removal, and lemmatization via [spaCy](https://spacy.io/) `es_core_news_sm` |
| **Classical ML baseline** | TF-IDF vectorization (top 5 000 features) + `LinearSVC` with balanced class weights |
| **Transformer fine-tuning** | `TFBertForSequenceClassification` fine-tuned on BETO (`dccuchile/bert-base-spanish-wwm-cased`) |
| **Class imbalance handling** | `compute_class_weight('balanced')` applied during BETO training; stratified 80/20 train-validation split |
| **Optimal decision threshold** | Grid search over [0.30, 0.70] to maximize F1 for the positive (real news) class |
| **Rich evaluation metrics** | `classification_report` (precision, recall, F1), AUC-ROC score |
| **Mixed-dataset ingestion** | Supports `.xlsx` and `.csv` sources with automatic column normalization |
| **Balanced dataset construction** | Combines multiple Spanish corpora, down-samples the majority class to a configurable target ratio |
| **Modular codebase** | Preprocessing, training, and evaluation are independent, reusable modules |

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                      main.py                          │
│  Orchestrator: load → preprocess → train → evaluate  │
└──────────┬────────────────────────┬───────────────────┘
           │                        │
           ▼                        ▼
┌─────────────────────┐   ┌─────────────────────────────┐
│  src/preprocessing  │   │   src/train_classic.py       │
│  ─────────────────  │   │  TF-IDF (5 000 features)    │
│  spaCy es_core_news │   │  + LinearSVC (balanced)      │
│  - Lemmatization    │   │  → src/evaluation.py         │
│  - Stop-word removal│   │    (classification_report,  │
│  - URL/noise strip  │   │     AUC-ROC)                 │
└─────────────────────┘   └─────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────────┐
                        │   src/train_transformer.py       │
                        │  BETO (bert-base-spanish-wwm)    │
                        │  TFBertForSequenceClassification │
                        │  - Adam (lr=2e-5), 5 epochs      │
                        │  - Class-weight correction       │
                        │  - 80/20 stratified split        │
                        │  → src/evaluation.py             │
                        │    (optimal threshold, AUC-ROC)  │
                        └─────────────────────────────────┘
```

### Data Flow

```
data/train.xlsx  ─┐
data/esp_fake_news.csv ─┤→ cargar_datos_mixtos() → balanced X_train / y_train
data/archive/onlytrue1000.csv  ─┤
data/archive/onlyfakes1000.csv ─┘

data/test.xlsx → X_test / y_test (held-out evaluation set)
```

### Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| NLP preprocessing | spaCy 3.x (`es_core_news_sm`) |
| Classical ML | scikit-learn (TF-IDF, LinearSVC, Pipeline) |
| Deep learning framework | TensorFlow 2.x / Keras |
| Pre-trained language model | Hugging Face Transformers — BETO |
| Data I/O | pandas, openpyxl |
| Model persistence | joblib |

---

## Non-Functional Requirements

### Security

| Requirement | Implementation |
|---|---|
| No secrets in source code | All file paths are parameterised function arguments; no API keys, tokens, or credentials are hardcoded anywhere in the codebase |
| Safe data handling | Input data is loaded read-only; no user-supplied strings are executed or evaluated |
| Dependency hygiene | All dependencies are version-pinned in `requirements.txt` to prevent supply-chain drift |
| Logging discipline | TensorFlow and Hugging Face verbose logging is suppressed to `ERROR` level (`logging.basicConfig(level=logging.ERROR)`) to avoid leaking internal state to standard output in production environments |

### High Availability

| Requirement | Implementation |
|---|---|
| Graceful degradation | Both `evaluar_modelo_clasico` and `evaluar_modelo_transformer` wrap execution in `try/except` blocks, ensuring the pipeline reports errors without crashing the entire run |
| Reproducible outputs | All random operations use a fixed `random_state=42` seed (scikit-learn splits, SVM, sampling), guaranteeing deterministic results across runs |
| Resilient data loading | `cargar_datos_mixtos` uses `os.path.join` for cross-platform path resolution; missing columns raise descriptive `KeyError` exceptions |
| spaCy model auto-recovery | `preprocessing.py` catches `OSError` on model load and automatically downloads `es_core_news_sm` if absent, eliminating a common first-run failure |
| Stateless pipeline design | No global mutable state is shared between modules; the pipeline can be re-invoked safely without side effects |
| Containerisation-ready | The codebase has no hard dependency on a specific OS, GPU, or file-system path structure, facilitating deployment in Docker or cloud environments |

---

## Project Structure

```
Fake-News-IA/
├── main.py                    # Pipeline entry point
├── requirements.txt           # Pinned Python dependencies
├── src/
│   ├── preprocessing.py       # Text cleaning, lemmatization, dataset loading
│   ├── train_classic.py       # TF-IDF + SVM pipeline definition and training
│   ├── train_transformer.py   # BETO fine-tuning (TensorFlow / Keras)
│   └── evaluation.py         # Metrics, AUC-ROC, optimal threshold search
└── data/
    ├── train.xlsx             # Primary training set (Excel)
    ├── test.xlsx              # Held-out test set (Excel)
    ├── esp_fake_news.csv      # Spanish fake news corpus (CSV)
    └── archive/
        ├── onlytrue1000.csv   # 1 000 real news samples
        ├── onlyfakes1000.csv  # 1 000 fake news samples
        ├── fakes1000.csv      # Additional fake news archive
        ├── train.csv          # Archive training split
        └── test.csv           # Archive test split
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or later
- `pip` (bundled with Python)
- Sufficient disk space for BETO model weights (~440 MB, downloaded automatically from Hugging Face Hub on first run)
- A CUDA-capable GPU is **recommended** for the Transformer training step but is not required (CPU training is supported)

### 1. Clone the repository

```bash
git clone https://github.com/Juanvelandia-p/Fake-News-IA.git
cd Fake-News-IA
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the spaCy Spanish language model

```bash
python -m spacy download es_core_news_sm
```

> **Note:** If `es_core_news_sm` is not found at runtime, `preprocessing.py` will attempt the download automatically. Performing it manually as above is preferred in headless or restricted environments.

### 5. Verify data files

Ensure the following files are present before running the pipeline:

```
data/train.xlsx
data/test.xlsx
data/esp_fake_news.csv
data/archive/onlytrue1000.csv
data/archive/onlyfakes1000.csv
```

---

## Environment Variables & Safe Configuration

This project does not require any environment variables for standard local operation. All runtime parameters (file paths, model hyperparameters) are defined as named constants or function arguments within the source modules.

| Parameter | Location | Default | Description |
|---|---|---|---|
| `MODEL_NAME` | `src/train_transformer.py` | `dccuchile/bert-base-spanish-wwm-cased` | Hugging Face model identifier for BETO |
| `MAX_LEN` | `src/train_transformer.py` | `128` | Maximum token sequence length |
| `BATCH_SIZE` | `src/train_transformer.py` | `32` | Training batch size |
| `LEARNING_RATE` | `src/train_transformer.py` | `2e-5` | Adam optimizer learning rate |
| `EPOCHS` | `src/train_transformer.py` | `5` | Number of training epochs |
| `max_features` | `main.py` | `5000` | Maximum vocabulary size for TF-IDF |

**Security note:** No API tokens, database credentials, or other secrets are used or stored anywhere in this repository. If you extend the project to serve predictions via an API, use environment variables (e.g., via `python-dotenv`) and **never** commit secret values to version control.

---

## Running the Pipeline

Execute the full end-to-end pipeline (data loading → preprocessing → SVM training & evaluation → BETO training & evaluation):

```bash
python main.py
```

### Expected output (abridged)

```
-> Cargando y balanceando datasets adicionales (1000 True / 1000 Fake)...
Total datos de Entrenamiento (Real: 1338, Falsa: 1500): 2838
Total datos de Prueba (Real: ..., Falsa: ...): ...
--------------------------------------------------
Pesos de Clase Calculados (0: Falsa, 1: Real): {0: ..., 1: ...}
--------------------------------------------------

-> Entrenando Clasificador Clásico (SVM) como baseline...

==================================================
          Resultados del Clasificador Clásico (SVM)
==================================================
              precision    recall  f1-score   support
...
AUC-ROC Score: 0.XXXX

-> Tokenizando datos de entrenamiento...
-> Entrenando modelo (BETO) con Pesos de Clase...
Epoch 1/5 ...
...
-> Umbral Óptimo Encontrado (maximizando F1 Clase 1): 0.XX (F1 = 0.XX)

==================================================
          Resultados del Modelo Transformer (BETO) - Umbral Optimizado
==================================================
...
AUC-ROC Score: 0.XXXX
```

### Running individual modules

```bash
# Run only preprocessing validation (interactive Python)
python -c "from src.preprocessing import cargar_datos_mixtos; print('Preprocessing module OK')"

# Run only SVM training (requires data files)
python -c "
from src.preprocessing import cargar_datos_mixtos, limpiar_texto
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = cargar_datos_mixtos('data/train.xlsx','data/esp_fake_news.csv','data/test.xlsx')
print('Data loaded:', len(X_train), 'train samples')
"
```

---

## Data

| File | Description | Format | Labels |
|---|---|---|---|
| `data/train.xlsx` | Primary training corpus (unbalanced) | Excel | `True` / `Fake` |
| `data/test.xlsx` | Held-out test set | Excel | `TRUE` / `FALSE` |
| `data/esp_fake_news.csv` | Spanish fake news corpus | CSV | All fake (`Fake`) |
| `data/archive/onlytrue1000.csv` | 1 000 real news items | CSV | All real (`True`) |
| `data/archive/onlyfakes1000.csv` | 1 000 fake news items | CSV | All fake (`Fake`) |

**Class balancing strategy:** The pipeline constructs a final training set of approximately 1,338 real + 1,500 fake articles (random seed 42) to achieve a near-1:1 ratio, mitigating majority-class bias.

---

## Testing & Coverage

The repository does not currently include an automated test suite. Contributions adding unit or integration tests are warmly welcomed (see [Contribution Guidelines](#contribution-guidelines)).

Suggested test targets for contributors:

- `src/preprocessing.limpiar_texto` — unit tests for edge cases (empty string, URL-only, numeric text).
- `src/preprocessing.cargar_datos_mixtos` — integration tests using small fixture CSV/Excel files.
- `src/evaluation.find_optimal_threshold` — property-based tests validating threshold range and F1 maximisation.

---

## Deployment

### Local batch inference

The current design is a **batch pipeline** intended for offline training and evaluation. To use trained models for inference on new data:

1. After running `main.py`, persist the SVM pipeline using `joblib` (see `src/train_classic.py`):
   ```python
   import joblib
   joblib.dump(pipeline_svm, 'models/modelo_svm.pkl')
   ```
2. Load and call `.predict()` on new article text:
   ```python
   import joblib
   model = joblib.load('models/modelo_svm.pkl')
   prediction = model.predict(["Texto del artículo a clasificar"])
   # Returns: [0] (Fake) or [1] (Real)
   ```

### Docker (recommended for production)

A `Dockerfile` is not yet included in this repository. A minimal example for containerising the pipeline:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download es_core_news_sm

COPY . .
CMD ["python", "main.py"]
```

Build and run:

```bash
docker build -t fake-news-ia .
docker run --rm -v $(pwd)/data:/app/data fake-news-ia
```

### Cloud / GPU

For accelerated BETO fine-tuning, deploy to any GPU-backed cloud environment (e.g., Google Colab, AWS SageMaker, GCP Vertex AI). TensorFlow will automatically detect and use available GPUs with no code changes required.

---

## CI/CD

The repository uses **GitHub Actions** for automated workflow orchestration.

### Active Workflows

| Workflow | Trigger | Description |
|---|---|---|
| **Copilot Coding Agent** | Issue / PR assignment to GitHub Copilot | Dynamic workflow that provisions a sandboxed environment and runs the Copilot SWE agent to implement code changes from issue descriptions |

### Planned Workflows (not yet implemented)

The following CI/CD stages are recommended for teams adopting this project in a professional setting:

| Stage | Suggested Trigger | Suggested Action |
|---|---|---|
| **Lint** | `push`, `pull_request` | `flake8` / `ruff` — enforce PEP 8 and code quality |
| **Unit tests** | `push`, `pull_request` | `pytest` with coverage report |
| **Data validation** | `push` to `main` | Assert dataset schema and label distributions |
| **Model training** | Manual / scheduled | Run `main.py` on a GPU runner and publish metrics |
| **Container build** | `push` to `main` | Build and push Docker image to registry |

Contributions adding any of the above workflows are encouraged (see [Contribution Guidelines](#contribution-guidelines)).

---

## Contribution Guidelines

We welcome contributions from professionals at all levels. Please follow these standards:

### Getting started

1. Fork the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-descriptive-feature-name
   ```
2. Set up the development environment following the [Setup & Installation](#setup--installation) instructions.
3. Make focused, atomic commits with clear messages (imperative mood: *"Add threshold optimisation test"*, not *"added stuff"*).

### Code standards

- **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/). Run `flake8 .` or `ruff check .` before submitting.
- **Docstrings:** All public functions must have a Google-style docstring describing purpose, arguments, and return values.
- **Type hints:** New functions should include type annotations where practical.
- **No secrets:** Never commit API keys, credentials, or personal data. Use environment variables.

### Pull request process

1. Ensure your branch is up to date with `main` before opening a PR.
2. Fill in the PR template with a description of the change, motivation, and testing performed.
3. PRs require at least one approving review before merge.
4. Squash merge is preferred to keep commit history clean.

### Reporting issues

- Use GitHub Issues to report bugs or request features.
- Include: Python version, OS, full traceback, and steps to reproduce.

---

## License

This repository does not currently include a license file. Please contact the repository owner before using or redistributing this code in commercial or production contexts.

---

*Built with ❤️ for Spanish-language NLP.*
