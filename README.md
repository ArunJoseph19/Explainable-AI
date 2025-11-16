# 🎨 Explainable AI: FLUX.1-Kontext Image Generation & Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for explainable AI-powered image generation using FLUX.1-Kontext. This project provides interactive dashboards for generating, analyzing, and understanding how diffusion models transform images, with built-in word attribution analysis and LLM-powered insights.

## 🌟 Features

- **🤖 AI-Powered Prompt Generation**: Automatically generate structured prompts from images using LangChain agents with safety filtering
- **🎨 FLUX.1-Kontext Integration**: Generate high-quality image variations for multiple scenarios (mugs, t-shirts, gift bags)
- **🔬 Word Attribution Analysis**: Visualize the impact of individual prompt words on generated images through ablation studies
- **📊 Interactive Dashboards**: 
  - Generator Dashboard: Create new images with real-time progress tracking
  - Analysis Dashboard: Review results, evolution timelines, and word impact rankings
- **🧠 LLM Analysis**: Get AI-powered feedback on generated images with suggestions for prompt improvements
- **⏱️ Diffusion Process Visualization**: Watch the step-by-step evolution of image generation

## 📁 Project Structure

```
Explainable-AI/
├── dashboards/              # Interactive Gradio web interfaces
│   ├── flux_generator_dashboard.py    # Image generation interface
│   ├── analysis_dashboard.py          # Results analysis interface
│   └── launch.py                       # Quick launcher script
├── src/                     # Source code
│   ├── agents/              # LangChain agents for prompt generation
│   │   ├── interact_agent.py          # Main agent with safety filtering
│   │   ├── context.py
│   │   ├── create_agent.py
│   │   └── ...
│   ├── core/                # Core utilities and tools
│   │   ├── retrievers.py
│   │   └── tools.py
│   └── flux/                # FLUX model utilities
│       ├── flux_server.py             # Flask API for FLUX generation
│       └── flux_logo_editor.py
├── notebooks/               # Jupyter notebooks for experimentation
│   ├── Demo.ipynb
│   ├── Flux Kontext Model.ipynb
│   ├── LLM Analysis.ipynb
│   └── ...
├── outputs/                 # Generated outputs and experiments
│   ├── experiments/flux_experiments/  # Timestamped experiment runs
│   ├── demo_outputs/
│   └── ...
├── scripts/                 # Utility scripts
├── config/                  # Configuration files
├── assets/                  # Static assets (images, logos)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for FLUX model)
- 16GB+ RAM
- 20GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ArunJoseph19/Explainable-AI.git
cd Explainable-AI
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API keys**

Create a `.env` file in the project root:
```bash
# AWS Bedrock (via Holistic AI)
HOLISTIC_AI_TEAM_ID=your_team_id
HOLISTIC_AI_API_TOKEN=your_api_token

# OpenAI (alternative)
OPENAI_API_KEY=your_openai_key

# Valyu Search
VALYU_API_KEY=your_valyu_key
```

### Running the Application

#### Option 1: Quick Launch (Interactive Menu)
```bash
python dashboards/launch.py
```

#### Option 2: Launch Dashboards Separately

**Generator Dashboard** (Port 7862):
```bash
python dashboards/flux_generator_dashboard.py
```

**Analysis Dashboard** (Port 7861):
```bash
python dashboards/analysis_dashboard.py
```

#### Option 3: Flask API Server
```bash
python src/flux/flux_server.py
```

## 📖 Usage Guide

### 1. Generate Images

1. Open the **Generator Dashboard** at `http://localhost:7862`
2. Click **"Load FLUX Model"** (one-time setup, ~2 minutes)
3. Upload an input image (logo or design)
4. Click **"Generate Prompt with AI"** to create an optimized prompt
5. Review the safety check results
6. Click **"Generate All Scenarios"** to create variations for:
   - Mug designs
   - T-shirt designs
   - Gift bag designs

The system automatically generates:
- Final output images
- Evolution grids (diffusion process)
- Word attribution visualizations
- Timestep snapshots

### 2. Analyze Results

1. Open the **Analysis Dashboard** at `http://localhost:7861`
2. The latest experiment loads automatically
3. Select a scenario and prompt variant
4. Review visualizations:
   - **Input/Output Comparison**: See before and after
   - **Word Attribution**: Understand which words matter most
   - **Evolution Timeline**: Scrub through generation steps
   - **Ablation Studies**: View images with specific words removed

5. Run LLM analysis:
   - Choose an LLM model (Claude recommended)
   - Click **"Analyze with LLM"**
   - Get detailed feedback on:
     - Prompt adherence
     - Logo preservation
     - Design quality
     - Word effectiveness
     - Improvement suggestions

## 🔬 Key Technologies

- **FLUX.1-Kontext-dev**: State-of-the-art image generation model
- **LangChain**: Agent orchestration and prompt engineering
- **Gradio**: Interactive web interfaces
- **AWS Bedrock**: Claude LLM access (via Holistic AI)
- **Diffusers**: Hugging Face diffusion model library
- **PyTorch**: Deep learning framework

## 📊 Experiment Output Structure

Each experiment run creates:
```
outputs/experiments/flux_experiments/run_YYYYMMDD_HHMMSS/
├── input_image.png
├── mug_design/
│   └── prompt_0_generated/
│       ├── final_output.png
│       ├── evolution_grid.png
│       ├── word_attribution_complete.png
│       ├── metadata.json
│       ├── snapshots/
│       │   ├── step_000_t1000.0.png
│       │   ├── step_007_t875.0.png
│       │   └── ...
│       ├── ablated_without_festive.png
│       └── llm_analysis/
│           └── claude_3.5_sonnet_analysis.json
├── tshirt_design/
└── giftbag_design/
```

## 🧪 Notebooks

Explore the `notebooks/` directory for:
- `Demo.ipynb`: Basic usage examples
- `Flux Kontext Model.ipynb`: Model exploration
- `LLM Analysis.ipynb`: LLM integration examples
- `Dashboard FLUX.ipynb`: Dashboard development

## 🔒 Safety Features

The system includes multi-layer content moderation:
1. **Keyword Filtering**: Blocks profanity and inappropriate content
2. **LLM Safety Check**: Claude evaluates prompts for safety concerns
3. **Structured Output Validation**: Ensures well-formed, safe prompts

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Black Forest Labs** for FLUX.1-Kontext-dev
- **Anthropic** for Claude models
- **Holistic AI** for Bedrock API access
- **Hugging Face** for model hosting and diffusers library

## 📧 Contact

**Arun Joseph**
- GitHub: [@ArunJoseph19](https://github.com/ArunJoseph19)
- Repository: [Explainable-AI](https://github.com/ArunJoseph19/Explainable-AI)

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce `MAX_RESOLUTION` in dashboards
- Enable more aggressive CPU offloading
- Close other GPU applications

**Model Loading Fails:**
- Check internet connection
- Verify Hugging Face authentication
- Ensure sufficient disk space

**API Errors:**
- Verify API keys in `.env`
- Check rate limits
- Review Holistic AI team quota

**Import Errors:**
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`
- Check Python version (3.8+ required)

## 🗺️ Roadmap

- [ ] Add support for more image formats
- [ ] Implement batch processing
- [ ] Add video generation capabilities
- [ ] Create REST API documentation
- [ ] Add Docker deployment
- [ ] Implement user authentication
- [ ] Add prompt library/templates

---

**Star ⭐ this repository if you find it useful!**
