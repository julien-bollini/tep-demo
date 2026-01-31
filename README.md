# 🏭 TEP Fault Detection & Diagnosis System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready **Machine Learning system** for real-time fault detection and diagnosis in the Tennessee Eastman Process (TEP) - a benchmark chemical reactor simulation used in process control research.

## 🎯 Overview

This project implements a **cascaded two-stage ML architecture** that:

1. **Detects** anomalies in reactor operation (Binary Classification)
2. **Diagnoses** the specific fault type among 20 possible faults (Multi-class Classification)

The system features a complete MLOps pipeline from data ingestion to deployment, with:
- 🚀 **FastAPI REST API** for real-time inference
- 📊 **Interactive Streamlit Dashboard** for monitoring
- 🐳 **Docker containerization** for easy deployment
- ⚙️ **Automated pipeline** with Makefile orchestration

---

## ✨ Key Features

### Machine Learning
- **Two-stage cascaded Random Forest models** (Detector → Diagnostician)
- **52 process variables** (41 measured + 11 manipulated sensors)
- **20 fault types** classification with high accuracy
- **Leakage-proof splitting** strategy (group-wise by simulation run)
- **Class imbalance handling** with balanced weights

### MLOps Pipeline
- **Medallion Architecture** (Bronze → Silver → Gold data layers)
- **Automated ETL** with Kaggle dataset integration
- **Model versioning** and artifact persistence
- **Idempotent operations** for reproducibility
- **Performance metrics** tracking and reporting

### Deployment
- **RESTful API** with health checks and error handling
- **Real-time streaming simulation** on dashboard
- **Reactor synoptic visualization** with LED indicators
- **Event detection** with configurable persistence thresholds
- **Docker Compose** multi-service orchestration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         ML Pipeline (main.py)               │
│   Preprocess → Train → Evaluate             │
└─────────────────┬───────────────────────────┘
                  │
                  ↓
         ┌────────────────┐
         │ Trained Models │
         │   (.joblib)    │
         └────────┬───────┘
                  │
        ┌─────────┴──────────┐
        ↓                    ↓
┌──────────────┐    ┌─────────────────┐
│  FastAPI     │    │  Streamlit      │
│  API Server  │←──→│  Dashboard      │
│  Port 8000   │    │  Port 8501      │
└──────────────┘    └─────────────────┘
```

### Two-Stage Model Architecture

**Stage 1: Detector (Binary Anomaly Detection)**
- Random Forest (50 estimators, max_depth=10)
- Classifies: Normal (0) vs Faulty (1)
- High sensitivity to catch all anomalies

**Stage 2: Diagnostician (Multi-class Fault Classification)**
- Random Forest (100 estimators, max_depth=20)
- Classifies: Fault 1-20
- Only triggered when anomaly detected
- Trained exclusively on faulty data

---

## 📁 Project Structure

```
tep-demo/
├── src/
│   ├── api/                   # FastAPI inference service
│   │   ├── main.py            # API endpoints & model loading
│   │   └── schemas.py         # Pydantic data validation
│   ├── dashboard/
│   │   └── app.py             # Streamlit monitoring UI
│   ├── preprocessing/
│   │   ├── downloader.py      # Kaggle data ingestion
│   │   └── processor.py       # ETL transformations
│   ├── training/
│   │   ├── loader.py          # Data loading & splitting
│   │   └── trainer.py         # Model training logic
│   ├── evaluation/
│   │   └── evaluator.py       # Performance metrics
│   └── config.py              # Global configuration
├── models/                    # Serialized ML artifacts
│   ├── detector_pipeline.joblib
│   ├── diagnostician_pipeline.joblib
│   └── metrics.json
├── data/                      # Data storage (Bronze/Silver/Gold)
│   ├── raw/                   # Original CSV files
│   ├── processed/             # Parquet files
│   └── final_split/           # Test set archive
├── docker-compose.yml         # Multi-service orchestration
├── Dockerfile.api             # API container
├── Dockerfile.dashboard       # Dashboard container
├── Makefile                   # Automation commands
├── requirements.txt           # Python dependencies
└── main.py                    # CLI pipeline orchestrator
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Make (optional, for automation)

### Platform-Specific Installation Guides

<details>
<summary><b>🐧 Fedora / RHEL / CentOS</b></summary>

#### 1. System Updates
```bash
# Update system packages
sudo dnf update -y
```

#### 2. Python 3.11+
```bash
# Install Python 3.11
sudo dnf install python3.11 python3.11-pip python3.11-devel -y

# Verify installation
python3.11 --version
```

#### 3. Development Tools
```bash
# Install essential build tools
sudo dnf groupinstall "Development Tools" -y

# Install additional dependencies
sudo dnf install gcc gcc-c++ make git -y
```

#### 4. Docker & Docker Compose
```bash
# Add Docker repository
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo

# Install Docker
sudo dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (avoid using sudo)
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
docker compose version
```

#### 5. Make (Optional but Recommended)
```bash
# Usually pre-installed, but if not:
sudo dnf install make -y
```

</details>

<details>
<summary><b>🐧 Ubuntu / Debian</b></summary>

#### 1. System Updates
```bash
# Update package lists and upgrade packages
sudo apt update && sudo apt upgrade -y
```

#### 2. Python 3.11+
```bash
# Add deadsnakes PPA for latest Python versions
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install pip
sudo apt install python3-pip -y

# Verify installation
python3.11 --version
```

#### 3. Development Tools
```bash
# Install build essentials
sudo apt install build-essential git curl wget -y
```

#### 4. Docker & Docker Compose
```bash
# Install prerequisites
sudo apt install ca-certificates curl gnupg lsb-release -y

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
docker compose version
```

#### 5. Make (Optional but Recommended)
```bash
# Usually pre-installed, but if not:
sudo apt install make -y
```

</details>

<details>
<summary><b>🍎 macOS</b></summary>

#### 1. Homebrew Package Manager
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Update Homebrew
brew update
```

#### 2. Python 3.11+
```bash
# Install Python 3.11
brew install python@3.11

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
python3.11 --version
```

#### 3. Development Tools
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Git (if not already installed)
brew install git
```

#### 4. Docker Desktop
```bash
# Install Docker Desktop via Homebrew
brew install --cask docker

# Or download manually from:
# https://www.docker.com/products/docker-desktop/

# Start Docker Desktop from Applications
# Verify installation
docker --version
docker compose version
```

#### 5. Make
```bash
# Usually pre-installed with Xcode CLI tools
# Verify
make --version
```

#### 6. Optional: Audio Notifications
```bash
# macOS has built-in 'say' command for audio feedback
# No additional installation needed
# Test: say "Hello from Terminal"
```

</details>

<details>
<summary><b>🪟 Windows</b></summary>

#### Windows Subsystem for Linux (WSL2) - Recommended Approach

**Step 1: Enable WSL**
```powershell
# Run PowerShell as Administrator
wsl --install

# Restart your computer
```

**Step 2: Install Ubuntu on WSL**
```powershell
# Install Ubuntu 22.04 LTS
wsl --install -d Ubuntu-22.04

# Launch Ubuntu and create user account
```

**Step 3: Follow Ubuntu instructions above**
- Once inside WSL, follow the Ubuntu/Debian installation steps
- All commands will work natively in WSL environment

</details>

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/jubenkai73/tep-demo
cd tep-demo

# Create virtual environment and install dependencies
make setup

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
# Execute full pipeline (data → training → evaluation)
make pipeline

# Or run individual steps:
make preprocess    # ETL pipeline
make train         # Model training
make evaluate      # Performance metrics
```

### Deploy with Docker

```bash
# Build and start services (API + Dashboard)
make deploy

# Or using docker-compose directly:
docker-compose up -d

# Access the dashboard
open http://localhost:8501
```

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Inference request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensors": {
      "xmeas_1": 0.25038,
      "xmeas_2": 3674.0,
      "xmeas_3": 4509.3,
      "xmeas_4": 9.3477,
      ...
      "xmv_11": 17.373
    }
  }'
```

---

## 📊 Dataset

### Tennessee Eastman Process (TEP)
- **Source**: [Kaggle - TEP CSV Dataset](https://www.kaggle.com/datasets/afrniomelo/tep-csv)
- **Domain**: Chemical engineering benchmark for process control
- **Features**: 52 process variables
  - 41 measured variables (xmeas_1 to xmeas_41)
  - 11 manipulated variables (xmv_1 to xmv_11)
- **Target**: 21 classes (0 = Normal, 1-20 = Fault types)
- **Temporal Resolution**: 3-minute sampling intervals
- **Simulation Length**: 500 samples per run (~25 hours)

### Key Variables
- **xmeas_7**: Reactor Pressure (kPa)
- **xmeas_9**: Reactor Temperature (°C)
- **xmeas_10**: Product Flow Rate (kg/hr)

---

## 🎮 Dashboard Usage

1. **Select Fault Scenario**: Choose from Fault 1-20 dropdown
2. **Start Simulation**: Click "▶️ Start Simulation" button
3. **Monitor Real-time**:
   - Reactor synoptic with LED status indicators
   - Time-series charts (Pressure, Temperature, Flow)
   - Fault diagnosis timeline
4. **Review Results**: Post-simulation report with detection/diagnosis delays

### Dashboard Features
- 🎨 **Reactor Synoptic**: Visual schematic with color-coded alerts
- 📈 **Time-series Charts**: Live updating Plotly visualizations
- ⏱️ **Event Timeline**: Tracks when faults are detected and diagnosed
- 📊 **Performance Metrics**: F1 scores and accuracy per fault type
- 🔔 **Stabilization Period**: First 60 minutes hidden to allow system warm-up

---

## ⚙️ Configuration

Key parameters in `src/config.py`:

```python
# Global Settings
RANDOM_SEED = 42                    # For reproducibility
DEFAULT_TEST_SIZE = 0.2             # 80/20 train-test split
DEFAULT_N_SIMULATIONS = None        # Full dataset (or limit for prototyping)

# Model Hyperparameters
DETECTOR_PARAMS = {
    "n_estimators": 50,
    "max_depth": 10,
    "class_weight": "balanced",
    "n_jobs": -1
}

DIAGNOSTICIAN_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "class_weight": "balanced",
    "n_jobs": -1
}
```

---

## 🛠️ Makefile Commands

```bash
make help          # Display all available commands

# Development
make setup         # Create venv and install dependencies
make lint          # Run Ruff linter with auto-fix
make pipeline      # Run complete ML pipeline

# Individual Pipeline Steps
make preprocess    # Data extraction and transformation
make train         # Train cascaded models
make evaluate      # Generate performance metrics

# Deployment
make deploy        # Build Docker images and start services
make build         # Build Docker images only
make up            # Start containers in detached mode
make down          # Stop and remove containers

# Maintenance
make clean         # Deep clean (venv, cache, models, artifacts)
make clean-cache   # Remove Kaggle cache only
make restart       # Restart Docker containers

# Force Operations
FORCE=true make pipeline      # Force full recomputation
make force-deploy             # Force metrics recompute + Docker rebuild
```

---

## 🔄 Data Pipeline

### Medallion Architecture

**Bronze Layer** (Raw Data)
- Download from Kaggle
- Store original CSV files
- Validate data integrity

**Silver Layer** (Processed)
- Convert CSV → Parquet (50% memory reduction)
- Apply optimized dtypes (float64 → float32)
- Temporal windowing (samples 140-639)
- Merge normal + faulty datasets

**Gold Layer** (Model-Ready)
- Group-wise train/test split (by simulation run)
- Feature/target separation
- Archive final test set for evaluation

### Pipeline Execution

```bash
python main.py --step preprocess  # ETL only
python main.py --step train       # Training only
python main.py --step evaluate    # Evaluation only
python main.py --step all         # Complete pipeline
python main.py --step train --force  # Force retrain
```

---

## 📈 Model Performance

The system tracks comprehensive metrics per fault:

- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual faults detected
- **F1-Score**: Harmonic mean of precision and recall
- **Global Accuracy**: Overall classification accuracy

Metrics are saved in `models/metrics.json` and displayed in:
- Terminal (formatted table via evaluator)
- Dashboard (metadata panel)

---

## 🐳 Docker Deployment

### Services

**API Service** (`Dockerfile.api`)
- Base: Python 3.11-slim
- Exposes: Port 8000
- Health check: `GET /health`
- Hot-reload: Enabled for development

**Dashboard Service** (`Dockerfile.dashboard`)
- Base: Python 3.11-slim
- Exposes: Port 8501
- Connects to: `http://api:8000`
- Volume mounts for live code updates

### Network
- Bridge network: `tep-network`
- Service discovery via DNS

### Environment Variables
- `API_URL`: Backend endpoint (default: `http://api:8000`)
- `PYTHONPATH`: Set to `/app`
- `ENV`: development/production

---

## 🧪 Testing

```bash
# Run tests (if test suite is implemented)
pytest tests/

# Test API endpoints
pytest tests/test_api.py

# Test data integrity
pytest tests/test_data_integrity.py

# Test model predictions
pytest tests/test_predictions.py
```

---

## 🔐 Best Practices Implemented

### MLOps
✅ Reproducible experiments (fixed random seed).
✅ Artifact versioning (joblib persistence).
✅ Data lineage tracking (cache metadata).
✅ Pipeline automation (Makefile + CLI).
✅ Idempotent operations (skip existing artifacts).

### Software Engineering
✅ Type hints throughout codebase
✅ Comprehensive docstrings
✅ DRY principle (centralized config)
✅ SOLID principles (single responsibility)
✅ API-first design (Pydantic schemas)

### Data Science
✅ Proper train/test splitting (group-wise)
✅ Feature standardization (StandardScaler)
✅ Class imbalance handling (balanced weights)
✅ Multi-stage architecture (reduces false positives)
✅ Evaluation metrics persistence

---

## 🚧 Future Enhancements

### Production
- [ ] Database integration (PostgreSQL/TimescaleDB)
- [ ] Model registry (MLflow/Weights & Biases)
- [ ] A/B testing framework
- [ ] Prometheus metrics + Grafana dashboards
- [ ] API authentication & rate limiting

### Machine Learning
- [ ] Feature engineering (rolling windows, lag features)
- [ ] Hyperparameter tuning (Optuna/GridSearch)
- [ ] Ensemble methods (XGBoost, LightGBM)
- [ ] Online learning capabilities
- [ ] SHAP values for explainability

### DevOps
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Kubernetes deployment
- [ ] Infrastructure as Code (Terraform)
- [ ] Secrets management (Vault)
- [ ] Load balancing (NGINX)

---

## 📚 Dependencies

### Core Libraries
```
pandas==2.2.3           # Data manipulation
scikit-learn==1.7.2     # ML models
fastapi==0.115.0        # REST API
streamlit==1.39.0       # Dashboard UI
plotly==5.24.1          # Visualizations
joblib==1.4.2           # Model serialization
pyarrow==17.0.0         # Parquet handling
kagglehub==0.3.3        # Dataset download
```

### Development Tools
```
ruff==0.6.9             # Linting & formatting
uvicorn==0.31.0         # ASGI server
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code passes `make lint`
- All tests pass
- Documentation is updated
- Commit messages are descriptive

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Tennessee Eastman Process Dataset**: Original benchmark from Downs & Vogel (1993)
- **Kaggle Community**: For hosting the TEP CSV dataset
- **FastAPI & Streamlit**: For excellent framework documentation
- **Scikit-learn**: For robust ML implementations

---

## 🛠️ Contribution & Bugs

Si vous trouvez un bug ou si vous avez une demande de fonctionnalité, merci d'ouvrir une **[Issue](https://github.com/jubenkai73/tep-demo/issues)**.

Pour les questions générales et l'entraide, rendez-vous plutôt dans l'onglet **[Discussions](https://github.com/jubenkai73/tep-demo/discussions)** !
---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

**Built with ❤️ for the MLOps community**
