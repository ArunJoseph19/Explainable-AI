#!/bin/bash

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Explainable AI - Setup Script                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate || { echo "❌ Failed to activate venv"; exit 1; }
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ pip upgraded"
echo ""

# Install requirements
echo "📥 Installing dependencies (this may take a few minutes)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "✅ Dependencies installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi
echo ""

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p outputs/experiments
mkdir -p config
mkdir -p assets
echo "✅ Directories created"
echo ""

# Check for .env file
echo "🔑 Checking for API keys..."
if [ -f ".env" ]; then
    echo "✅ .env file found"
else
    echo "⚠️  No .env file found. Creating template..."
    cat > .env << EOF
# AWS Bedrock (via Holistic AI)
HOLISTIC_AI_TEAM_ID=your_team_id_here
HOLISTIC_AI_API_TOKEN=your_api_token_here

# OpenAI (alternative)
OPENAI_API_KEY=your_openai_key_here

# Valyu Search
VALYU_API_KEY=your_valyu_key_here
EOF
    echo "✅ Template .env created. Please fill in your API keys."
fi
echo ""

# Check GPU availability
echo "🖥️  Checking GPU availability..."
python3 -c "import torch; print('✅ GPU Available:', torch.cuda.is_available())" 2>/dev/null || echo "⚠️  PyTorch not installed or GPU not available"
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  Setup Complete! 🎉                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys (if not done already)"
echo "2. Launch the application:"
echo "   python dashboards/launch.py"
echo ""
echo "Or run dashboards individually:"
echo "   python dashboards/flux_generator_dashboard.py"
echo "   python dashboards/analysis_dashboard.py"
echo ""
echo "For more information, see README.md"
echo ""
