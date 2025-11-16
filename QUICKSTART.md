# 🚀 Quick Start Guide

## Initial Setup (One-time)

```bash
# 1. Clone the repository
git clone https://github.com/ArunJoseph19/Explainable-AI.git
cd Explainable-AI

# 2. Run automated setup
./setup.sh

# 3. Configure API keys
nano .env  # Edit with your actual keys
```

## Running the Application

### Method 1: Interactive Launcher (Recommended)
```bash
python dashboards/launch.py
```
Then select from the menu:
- `1` - Start Generator Dashboard
- `2` - Start Analysis Dashboard
- `3` - Instructions for running both

### Method 2: Direct Launch

**Generator Dashboard:**
```bash
python dashboards/flux_generator_dashboard.py
# Opens at http://localhost:7862
```

**Analysis Dashboard:**
```bash
python dashboards/analysis_dashboard.py
# Opens at http://localhost:7861
```

## Basic Workflow

### 1. Generate Images
1. Open Generator Dashboard → `http://localhost:7862`
2. Load FLUX Model (first time only)
3. Upload input image
4. Generate AI prompt
5. Generate all scenarios
6. Copy the experiment path shown

### 2. Analyze Results
1. Open Analysis Dashboard → `http://localhost:7861`
2. Select scenario and prompt variant
3. Review visualizations
4. Run LLM analysis for insights

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Check GPU status
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Run Flask API server
python src/flux/flux_server.py
```

## Directory Structure Quick Reference

```
dashboards/     → Web interfaces
src/           → Source code
  ├── agents/  → LangChain agents
  ├── core/    → Utilities
  └── flux/    → FLUX model code
notebooks/     → Jupyter experiments
outputs/       → Generated results
scripts/       → Helper scripts
```

## Troubleshooting

**Out of Memory:**
- Close other applications
- Reduce image resolution
- Enable CPU offloading

**Import Errors:**
```bash
# Reinstall from project root
pip install -r requirements.txt --force-reinstall
```

**API Issues:**
- Check `.env` file has correct keys
- Verify API quota/rate limits
- Test with simple request first

## Need Help?

- Read full [README.md](README.md)
- Check [CONTRIBUTING.md](CONTRIBUTING.md)
- Open a GitHub Issue
- Review [notebooks/](notebooks/) for examples

---

**Happy Generating! 🎨**
