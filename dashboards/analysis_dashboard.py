"""
FLUX.1-Kontext Explainability Web Dashboard
Standalone web interface with Gradio - Auto-loads latest experiment
"""

import gradio as gr
from pathlib import Path
import json
import base64
import io
from PIL import Image
import requests
import time
import numpy as np

# ===== CONFIGURATION =====
OUTPUT_ROOT = Path("outputs/experiments")

# Auto-detect latest experiment
def get_latest_experiment():
    """Find the most recent experiment folder"""
    if not OUTPUT_ROOT.exists():
        return None

    experiment_folders = sorted(OUTPUT_ROOT.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if experiment_folders:
        return experiment_folders[0]
    return None

EXPERIMENT_ROOT = get_latest_experiment()

if EXPERIMENT_ROOT:
    print(f"📂 Loaded latest experiment: {EXPERIMENT_ROOT}")
else:
    print("⚠️ No experiments found in flux_experiments/")

# AWS Bedrock Configuration
API_ENDPOINT = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
TEAM_ID = "team_the_great_hack_2025_022"
API_TOKEN = "znqXT5zCmCynAx-kyx_hldrxvSeyaWvxzx55vB5mfNg"

LLMS_AVAILABLE = {
    "Claude 3.5 Sonnet (Recommended)": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Claude 4.5 Sonnet (Latest)": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "Claude 4 Opus (Powerful)": "us.anthropic.claude-opus-4-20250514-v1:0",
    "Claude 3 Opus": "us.anthropic.claude-3-opus-20240229-v1:0",
}

# ===== UTILITY FUNCTIONS =====

def resize_image_for_api(image_path, max_dimension=1200, quality=80):
    """Resize and encode image for Bedrock API"""
    with Image.open(image_path) as img:
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        width, height = img.size
        if max(width, height) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)

        img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return img_b64, 'image/jpeg'

def call_bedrock_llm(model_id, prompt, images=None):
    """Call AWS Bedrock API"""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_TOKEN
    }

    content = []
    if images:
        for img_path in images:
            try:
                img_b64, media_type = resize_image_for_api(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64
                    }
                })
            except Exception as e:
                return f"ERROR: Failed to encode {img_path}: {str(e)}"

    content.append({"type": "text", "text": prompt})

    payload = {
        "team_id": TEAM_ID,
        "api_token": API_TOKEN,
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2000,
        "temperature": 0.3
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=90)
        if response.status_code != 200:
            return f"ERROR: HTTP {response.status_code} - {response.text[:300]}"

        result = response.json()
        if "content" in result and len(result["content"]) > 0:
            return result["content"][0]["text"]
        return result.get("completion", "No response")

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"

def generate_analysis_prompt(prompt_text, scenario_name):
    """Generate LLM analysis prompt"""
    return f"""You are an expert in AI image generation and prompt engineering for diffusion models.

**Task**: Analyze this logo transformation for {scenario_name} design.

**Prompt used**: "{prompt_text}"
**Model**: FLUX.1-Kontext-dev

**Images provided**:
1. Input logo (original)
2. Generated output (transformed design)
3. Word attribution (3 rows: WITH word, WITHOUT word, difference heatmap)
4. Evolution grid (diffusion timesteps)

**Provide concise analysis**:

### 1. Prompt Adherence
Did the model follow instructions? What elements match/miss the prompt?

### 2. Logo Preservation
Is the original logo recognizable? Brand elements preserved?

### 3. Design Quality
Would this work for merchandise? Print quality concerns?

### 4. Word Attribution Insights
Which words (from attribution visualization) had the most impact? Which words are redundant or ineffective?

### 5. Prompt Improvements
Provide 3 specific, actionable changes to improve the output. Reference the word attribution results to explain WHY these changes will help.

**Format**: Use clear headings and bullet points. Be specific and actionable."""

def get_experiment_structure():
    """Scan experiment folder and return structure"""
    structure = {}
    if not EXPERIMENT_ROOT or not EXPERIMENT_ROOT.exists():
        return structure

    for scenario_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if scenario_dir.is_dir() and not scenario_dir.name.startswith('.'):
            scenario_name = scenario_dir.name
            structure[scenario_name] = []

            for prompt_dir in sorted(scenario_dir.iterdir()):
                if prompt_dir.is_dir() and prompt_dir.name.startswith("prompt_"):
                    metadata_file = prompt_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)

                        structure[scenario_name].append({
                            "name": prompt_dir.name,
                            "path": prompt_dir,
                            "prompt": metadata.get("prompt", "N/A"),
                            "metadata": metadata
                        })

    return structure

def calculate_word_impact(prompt_dir):
    """Calculate impact scores from ablated images"""
    baseline_path = prompt_dir / "final_output.png"
    if not baseline_path.exists():
        return {}

    baseline = np.array(Image.open(baseline_path).convert('RGB')).astype(float)

    ablated_files = list(prompt_dir.glob("ablated_without_*.png"))
    impacts = {}

    for ablated_path in ablated_files:
        word = ablated_path.stem.replace("ablated_without_", "")
        ablated = np.array(Image.open(ablated_path).convert('RGB')).astype(float)

        diff = np.linalg.norm(baseline - ablated, axis=2)
        impact_score = diff.mean()

        impacts[word] = {
            "score": impact_score,
            "path": str(ablated_path)
        }

    return dict(sorted(impacts.items(), key=lambda x: x[1]["score"], reverse=True))

# ===== GRADIO FUNCTIONS =====

def refresh_experiments():
    """Refresh experiment list"""
    global EXPERIMENT_ROOT, experiment_data
    EXPERIMENT_ROOT = get_latest_experiment()
    experiment_data = get_experiment_structure()

    if EXPERIMENT_ROOT:
        return f"✅ Loaded: {EXPERIMENT_ROOT.name}", gr.Dropdown(choices=list(experiment_data.keys()), value=list(experiment_data.keys())[0] if experiment_data else None)
    else:
        return "⚠️ No experiments found", gr.Dropdown(choices=[])

experiment_data = get_experiment_structure()

def get_prompts_for_scenario(scenario):
    """Get prompt options for selected scenario"""
    if not scenario or scenario not in experiment_data:
        return gr.Dropdown(choices=[])

    prompts = experiment_data[scenario]
    choices = [(f"{p['name']}: {p['prompt'][:60]}...", p['name']) for p in prompts]
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

def load_visualization(scenario, prompt_name):
    """Load all visualizations for selected prompt"""
    if not scenario or not prompt_name:
        return None, None, None, None, None, "⚠️ Please select a scenario and prompt"

    prompt_data = None
    for p in experiment_data[scenario]:
        if p['name'] == prompt_name:
            prompt_data = p
            break

    if not prompt_data:
        return None, None, None, None, None, "❌ Prompt not found"

    prompt_path = prompt_data['path']
    prompt_text = prompt_data['prompt']

    input_img = str(prompt_path / "input_image.png") if (prompt_path / "input_image.png").exists() else None
    output_img = str(prompt_path / "final_output.png") if (prompt_path / "final_output.png").exists() else None
    attr_img = str(prompt_path / "word_attribution_complete.png") if (prompt_path / "word_attribution_complete.png").exists() else None
    evolution_img = str(prompt_path / "evolution_grid.png") if (prompt_path / "evolution_grid.png").exists() else None

    impacts = calculate_word_impact(prompt_path)
    impact_text = "### 📊 Word Impact Rankings\n\n"
    for i, (word, data) in enumerate(impacts.items(), 1):
        impact_text += f"**#{i} \"{word}\"** - Impact Score: {data['score']:.2f}\n\n"

    if not impacts:
        impact_text = "⚠️ No word impact data available"

    info_text = f"""## 📝 Current Prompt

**{prompt_text}**

---

{impact_text}
"""

    return input_img, output_img, attr_img, evolution_img, info_text, "✅ Visualization loaded"

def get_evolution_snapshots(scenario, prompt_name):
    """Get list of evolution snapshots for timeline"""
    if not scenario or not prompt_name:
        return gr.Slider(maximum=0), None

    prompt_data = None
    for p in experiment_data[scenario]:
        if p['name'] == prompt_name:
            prompt_data = p
            break

    if not prompt_data:
        return gr.Slider(maximum=0), None

    snapshot_dir = prompt_data['path'] / "snapshots"
    if not snapshot_dir.exists():
        return gr.Slider(maximum=0), None

    snapshots = sorted(snapshot_dir.glob("step_*.png"))
    if not snapshots:
        return gr.Slider(maximum=0), None

    return gr.Slider(maximum=len(snapshots)-1, value=0), str(snapshots[0])

def show_evolution_step(scenario, prompt_name, step_index):
    """Show specific evolution step"""
    if not scenario or not prompt_name:
        return None

    prompt_data = None
    for p in experiment_data[scenario]:
        if p['name'] == prompt_name:
            prompt_data = p
            break

    if not prompt_data:
        return None

    snapshot_dir = prompt_data['path'] / "snapshots"
    if not snapshot_dir.exists():
        return None

    snapshots = sorted(snapshot_dir.glob("step_*.png"))
    if 0 <= step_index < len(snapshots):
        return str(snapshots[step_index])

    return None

def get_ablated_images(scenario, prompt_name):
    """Get ablated images for display"""
    if not scenario or not prompt_name:
        return [None] * 4

    prompt_data = None
    for p in experiment_data[scenario]:
        if p['name'] == prompt_name:
            prompt_data = p
            break

    if not prompt_data:
        return [None] * 4

    impacts = calculate_word_impact(prompt_data['path'])
    ablated_imgs = []

    for word, data in list(impacts.items())[:4]:
        ablated_imgs.append(str(data['path']))

    while len(ablated_imgs) < 4:
        ablated_imgs.append(None)

    return ablated_imgs

def analyze_with_llm(scenario, prompt_name, llm_choice, progress=gr.Progress()):
    """Run LLM analysis"""
    if not scenario or not prompt_name:
        return "⚠️ Please select a scenario and prompt first"

    progress(0, desc="Finding prompt data...")

    prompt_data = None
    for p in experiment_data[scenario]:
        if p['name'] == prompt_name:
            prompt_data = p
            break

    if not prompt_data:
        return "❌ Prompt not found"

    prompt_path = prompt_data['path']
    prompt_text = prompt_data['prompt']
    scenario_name = scenario.replace('_', ' ')

    progress(0.2, desc="Collecting images...")

    images_to_analyze = []
    for img_name in ["input_image.png", "final_output.png", 
                      "word_attribution_complete.png", "evolution_grid.png"]:
        img_path = prompt_path / img_name
        if img_path.exists():
            images_to_analyze.append(str(img_path))

    progress(0.4, desc="Generating analysis prompt...")

    analysis_prompt = generate_analysis_prompt(prompt_text, scenario_name)
    model_id = LLMS_AVAILABLE[llm_choice]

    progress(0.5, desc=f"Calling {llm_choice}... (this may take 30-60s)")

    start_time = time.time()
    response = call_bedrock_llm(model_id, analysis_prompt, images_to_analyze)
    duration = time.time() - start_time

    progress(0.9, desc="Saving results...")

    if not response.startswith("ERROR"):
        output_dir = prompt_path / "llm_analysis"
        output_dir.mkdir(exist_ok=True)

        llm_key = llm_choice.replace(" ", "_").replace("(", "").replace(")", "").lower()
        result = {
            "llm": llm_choice,
            "model_id": model_id,
            "prompt": prompt_text,
            "analysis": response,
            "response_time_seconds": duration,
            "timestamp": time.time()
        }

        with open(output_dir / f"{llm_key}_analysis.json", "w") as f:
            json.dump(result, f, indent=2)

        progress(1.0, desc="Complete!")

        return f"""✅ **Analysis Complete** (took {duration:.1f}s)

---

{response}
"""
    else:
        return f"❌ **Analysis Failed**\n\n{response}"

# ===== BUILD GRADIO INTERFACE =====

with gr.Blocks(title="FLUX.1-Kontext Explainability Dashboard", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0; font-size: 2.5em;'>🎨 FLUX.1-Kontext Explainability Dashboard</h1>
        <p style='color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.2em;'>
            Interactive Analysis of Diffusion Model Image Generation
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🎯 Navigation")

            refresh_btn = gr.Button("🔄 Refresh Experiments", size="sm")
            experiment_status = gr.Markdown(f"📂 Current: {EXPERIMENT_ROOT.name if EXPERIMENT_ROOT else 'None'}")

            scenario_dropdown = gr.Dropdown(
                choices=list(experiment_data.keys()),
                label="Select Scenario",
                value=list(experiment_data.keys())[0] if experiment_data else None
            )

            prompt_dropdown = gr.Dropdown(
                choices=[],
                label="Select Prompt Variant"
            )

            load_button = gr.Button("📊 Load Visualization", variant="primary", size="lg")

            status_text = gr.Markdown("ℹ️ Select a scenario and prompt to begin")

            gr.Markdown("---")
            gr.Markdown("## 🤖 LLM Analysis")

            llm_dropdown = gr.Dropdown(
                choices=list(LLMS_AVAILABLE.keys()),
                label="Select LLM Model",
                value="Claude 3.5 Sonnet (Recommended)"
            )

            analyze_button = gr.Button("🔍 Analyze with LLM", variant="success", size="lg")

        with gr.Column(scale=3):
            gr.Markdown("## 📸 Input → Output Comparison")

            with gr.Row():
                input_image = gr.Image(label="🎯 Input Logo", height=300)
                output_image = gr.Image(label="✨ Generated Output", height=300)

            info_markdown = gr.Markdown("Select a prompt to view details")

            gr.Markdown("## 🔬 Word Attribution Analysis")
            gr.Markdown("*Shows impact of individual words: WITH word (top), WITHOUT word (middle), Difference heatmap (bottom)*")
            attribution_image = gr.Image(label="Word Attribution Visualization", height=400)

            gr.Markdown("## 🖼️ Ablated Images (Top Impact Words)")
            with gr.Row():
                ablated_1 = gr.Image(label="Word #1 Removed", height=200)
                ablated_2 = gr.Image(label="Word #2 Removed", height=200)
                ablated_3 = gr.Image(label="Word #3 Removed", height=200)
                ablated_4 = gr.Image(label="Word #4 Removed", height=200)

            gr.Markdown("## ⏱️ Diffusion Process Evolution")
            evolution_image = gr.Image(label="Evolution Grid", height=400)

            gr.Markdown("## 🎬 Interactive Timeline")
            timeline_slider = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                value=0,
                label="Scrub through denoising steps"
            )
            timeline_image = gr.Image(label="Current Step", height=350)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📝 LLM Analysis Results")
            analysis_output = gr.Markdown("Click 'Analyze with LLM' to generate analysis")

    # Event handlers
    refresh_btn.click(
        fn=refresh_experiments,
        outputs=[experiment_status, scenario_dropdown]
    )

    scenario_dropdown.change(
        fn=get_prompts_for_scenario,
        inputs=[scenario_dropdown],
        outputs=[prompt_dropdown]
    )

    load_button.click(
        fn=load_visualization,
        inputs=[scenario_dropdown, prompt_dropdown],
        outputs=[input_image, output_image, attribution_image, evolution_image, info_markdown, status_text]
    ).then(
        fn=get_evolution_snapshots,
        inputs=[scenario_dropdown, prompt_dropdown],
        outputs=[timeline_slider, timeline_image]
    ).then(
        fn=get_ablated_images,
        inputs=[scenario_dropdown, prompt_dropdown],
        outputs=[ablated_1, ablated_2, ablated_3, ablated_4]
    )

    timeline_slider.change(
        fn=show_evolution_step,
        inputs=[scenario_dropdown, prompt_dropdown, timeline_slider],
        outputs=[timeline_image]
    )

    analyze_button.click(
        fn=analyze_with_llm,
        inputs=[scenario_dropdown, prompt_dropdown, llm_dropdown],
        outputs=[analysis_output]
    )

# ===== LAUNCH =====
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from generator
        share=True,
        debug=True
    )
