"""
FLUX.1-Kontext Interactive Generation Dashboard
Combines prompt generation with FLUX image generation and analysis
"""

import gradio as gr
import torch
from diffusers import FluxKontextPipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import json
from datetime import datetime
import time
import base64
import io
import os
import sys

# Import the agent from interact_agent.py
try:
    from interact_agent import agent, config
    AGENT_AVAILABLE = True
    print("✅ Agent loaded successfully")
except Exception as e:
    print(f"⚠️ Agent not available: {e}")
    AGENT_AVAILABLE = False

# ===== CONFIGURATION =====
SCENARIOS = {
    "mug_design": {
        "base_prompt": "transform logo into festive holiday mug design with snowflakes",
        "variants": [
            "add christmas-themed mug design with candy canes and holly",
            "create winter coffee cup graphics with red and green accents",
            "design festive drinkware pattern with ornaments and ribbons"
        ]
    },
    "tshirt_design": {
        "base_prompt": "adapt logo for holiday t-shirt print with seasonal elements",
        "variants": [
            "create christmas apparel design with vintage holiday motifs",
            "transform into festive clothing graphic with snowflakes and stars",
            "design holiday wearable print with cozy winter theme"
        ]
    },
    "giftbag_design": {
        "base_prompt": "convert logo to christmas gift bag design with wrapping elements",
        "variants": [
            "create holiday gift wrap pattern with bows and ornaments",
            "design festive packaging graphics with presents and ribbons",
            "transform into christmas gift bag artwork with winter scenes"
        ]
    }
}

OUTPUT_ROOT = "flux_experiments"
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
MAX_RESOLUTION = 768
SKIP_WORDS = {'a', 'an', 'the', 'to', 'with', 'and', 'or', 'for', 'of', 'in', 'into'}

# Global state
FLUX_PIPE = None
CURRENT_EXPERIMENT_PATH = None

# ===== HELPER FUNCTIONS (from original code) =====

def unpack_flux_latents(latents):
    """Unpack FLUX latents from [B, seq_len, hidden_dim] to [B, C, H, W]"""
    batch_size = latents.shape[0]
    seq_len = latents.shape[1]
    hidden_dim = latents.shape[2]

    patch_size = int(seq_len ** 0.5)
    latent_channels = 16

    latents = latents.reshape(batch_size, patch_size, patch_size, hidden_dim)
    latents = latents.reshape(
        batch_size, patch_size, patch_size, latent_channels, 
        hidden_dim // latent_channels
    )
    latents = latents[..., 0]
    latents = latents.permute(0, 3, 1, 2).contiguous()

    return latents

snapshot_info = []
output_dir = None
current_step_callback = None

def decode_callback(pipe_obj, step_index, timestep, callback_kwargs):
    """Decode and save intermediate latents with live preview"""
    global output_dir, snapshot_info, current_step_callback

    if step_index % 7 != 0 and step_index != 0:
        return callback_kwargs

    try:
        latents = callback_kwargs["latents"]
        snapshot_dir = Path(output_dir) / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)

        unpacked_latents = unpack_flux_latents(latents)

        with torch.no_grad():
            decoded = pipe_obj.vae.decode(
                unpacked_latents / pipe_obj.vae.config.scaling_factor,
                return_dict=False
            )

            if isinstance(decoded, tuple):
                image_tensor = decoded[0]
            else:
                image_tensor = decoded

            image = (image_tensor / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            image = (image * 255).astype(np.uint8)

            filepath = snapshot_dir / f"step_{step_index:03d}_t{timestep:.1f}.png"
            Image.fromarray(image).save(filepath)
            snapshot_info.append((step_index, timestep))

            # Callback for live preview
            if current_step_callback:
                current_step_callback(image, step_index, timestep)

            del unpacked_latents, image_tensor, image, decoded
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ⚠️ Failed at step {step_index}: {e}")

    return callback_kwargs

def select_important_words(prompt, max_words=4):
    """Select the most important/descriptive words for ablation"""
    words = prompt.split()
    important_words = [w for w in words if w.lower() not in SKIP_WORDS]

    if len(important_words) > max_words:
        important_words = sorted(important_words, key=len, reverse=True)[:max_words]

    return important_words

def create_evolution_grid(output_dir):
    """Create diffusion process evolution grid"""
    snapshot_dir = Path(output_dir) / "snapshots"
    snapshot_files = sorted(snapshot_dir.glob("step_*.png"))

    if len(snapshot_files) == 0:
        return

    images = [Image.open(f) for f in snapshot_files]
    n = len(images)

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (img, file) in enumerate(zip(images, snapshot_files)):
        axes[i].imshow(img)
        step_info = file.stem.replace("step_", "Step ").replace("_t", " | t=")
        axes[i].set_title(step_info, fontsize=12)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "evolution_grid.png", dpi=120, bbox_inches='tight')
    plt.close()

def create_word_attribution_ablation(pipe, image_path, prompt, output_dir):
    """Generate word attribution ablation analysis"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_image = Image.open(image_path).convert("RGB")
    if max(input_image.size) > 768:
        ratio = 768 / max(input_image.size)
        new_size = tuple(int(dim * ratio // 16 * 16) for dim in input_image.size)
        input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)

    print("    Generating baseline for ablation...")
    torch.cuda.empty_cache()
    with torch.no_grad():
        baseline = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=20,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(42)
        )
    baseline_img = np.array(baseline.images[0]).astype(float)

    important_words = select_important_words(prompt, max_words=4)
    word_data = {}

    for word in important_words:
        print(f"      Ablating '{word}'...")
        ablated_prompt = prompt.replace(word, "").replace("  ", " ").strip()

        torch.cuda.empty_cache()
        with torch.no_grad():
            ablated = pipe(
                prompt=ablated_prompt,
                image=input_image,
                num_inference_steps=20,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator("cuda").manual_seed(42)
            )

        ablated_img = np.array(ablated.images[0]).astype(float)
        spatial_diff = np.linalg.norm(baseline_img - ablated_img, axis=2)

        word_data[word] = {
            'ablated_image': ablated_img,
            'heatmap': spatial_diff
        }

        del ablated
        torch.cuda.empty_cache()

    # Create visualization
    n_words = len(word_data)
    fig, axes = plt.subplots(3, n_words, figsize=(4*n_words, 12))
    if n_words == 1:
        axes = axes.reshape(3, 1)

    for i, (word, data) in enumerate(word_data.items()):
        axes[0, i].imshow(baseline_img.astype(np.uint8))
        axes[0, i].set_title(f'WITH "{word}"', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')

        axes[1, i].imshow(data['ablated_image'].astype(np.uint8))
        axes[1, i].set_title(f'WITHOUT "{word}"', fontsize=11, fontweight='bold', color='red')
        axes[1, i].axis('off')

        im = axes[2, i].imshow(data['heatmap'], cmap='hot', interpolation='bilinear')
        axes[2, i].set_title(f'Difference Map', fontsize=10)
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)

    plt.suptitle(f'Word Attribution Analysis\nPrompt: "{prompt}"', 
                 fontsize=13, y=0.99, fontweight='bold')

    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    plt.savefig(Path(output_dir) / "word_attribution_complete.png", dpi=120, bbox_inches='tight')
    plt.close()

    for word, data in word_data.items():
        ablated_pil = Image.fromarray(data['ablated_image'].astype(np.uint8))
        ablated_pil.save(Path(output_dir) / f"ablated_without_{word}.png")

    return word_data

# ===== GRADIO FUNCTIONS =====

def load_flux_model(progress=gr.Progress()):
    """Load FLUX model once"""
    global FLUX_PIPE

    if FLUX_PIPE is not None:
        return "✅ Model already loaded!"

    try:
        progress(0, desc="Loading FLUX.1-Kontext-dev model...")

        FLUX_PIPE = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16
        )

        progress(0.5, desc="Enabling optimizations...")
        FLUX_PIPE.enable_sequential_cpu_offload()
        FLUX_PIPE.enable_attention_slicing(1)
        FLUX_PIPE.enable_vae_slicing()

        progress(1.0, desc="Model loaded!")
        return "✅ FLUX model loaded successfully!"

    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def generate_prompt_from_image(image, progress=gr.Progress()):
    """Generate prompt using interact_agent.py"""

    if not AGENT_AVAILABLE:
        return None, "❌ Agent not available. Make sure interact_agent.py is in the same directory."

    if image is None:
        return None, "⚠️ Please upload an image first"

    try:
        progress(0, desc="Encoding image...")

        # Save temp image
        temp_path = Path("temp_input.png")
        Image.fromarray(image).save(temp_path)

        # Encode image to base64
        with open(temp_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode('utf-8')

        data_uri = f"data:image/png;base64,{base64_data}"

        progress(0.3, desc="Calling agent to analyze image...")

        # Call agent
        response = agent.invoke(
            {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate an image prompt based on this image."},
                        {"type": "image_url", "imageurl": {"url": data_uri}}
                    ]
                }]
            },
            config=config
        )

        progress(0.7, desc="Parsing response...")

        # Extract JSON from response
        json_string = response["messages"][-1].content.strip()
        json_string = json_string.removeprefix("```json").removesuffix("```").strip()

        data = json.loads(json_string)

        progress(0.9, desc="Validating safety...")

        # Check safety
        if not data.get("safe", False):
            return None, "❌ **Safety Check Failed**: Content flagged as unsafe by the agent."

        # Extract prompt string (from subject.main)
        prompt_obj = data.get("prompt", {})
        subject = prompt_obj.get("subject", {})
        main_prompt = subject.get("main", "")

        # Build full prompt from details
        details = subject.get("details", [])
        actions = subject.get("actions", [])

        full_prompt = main_prompt
        if details:
            full_prompt += ", " + ", ".join(details)
        if actions:
            full_prompt += ", " + ", ".join(actions)

        # Add environment
        env = prompt_obj.get("environment", {})
        if env.get("setting"):
            full_prompt += f", {env['setting']}"

        # Safety info
        safety_info = f"""### ✅ Safety Check Passed

**Content Moderation**: Safe
**Profanity Filter**: Passed
        """

        progress(1.0, desc="Complete!")

        # Clean up
        temp_path.unlink(missing_ok=True)

        return full_prompt, safety_info

    except Exception as e:
        return None, f"❌ Error generating prompt: {str(e)}"

def run_flux_generation(input_image, generated_prompt, progress=gr.Progress()):
    """Run complete FLUX generation pipeline"""
    global FLUX_PIPE, CURRENT_EXPERIMENT_PATH, output_dir, snapshot_info, current_step_callback

    if FLUX_PIPE is None:
        return None, None, None, None, "❌ Please load the model first!", None

    if input_image is None:
        return None, None, None, None, "⚠️ Please upload an image", None

    if not generated_prompt:
        return None, None, None, None, "⚠️ Please generate a prompt first", None

    try:
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_root = Path(OUTPUT_ROOT) / f"run_{timestamp}"
        CURRENT_EXPERIMENT_PATH = experiment_root

        # Save input image
        input_path = experiment_root / "input_image.png"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(input_image).save(input_path)

        # Storage for live previews
        live_snapshots = []

        def step_callback(image, step, timestep):
            live_snapshots.append((image, step, timestep))

        current_step_callback = step_callback

        total_scenarios = len(SCENARIOS)
        scenario_counter = 0

        results = {}

        for scenario_name, scenario_config in SCENARIOS.items():
            scenario_counter += 1
            base_progress = (scenario_counter - 1) / total_scenarios

            progress(base_progress, desc=f"Scenario {scenario_counter}/{total_scenarios}: {scenario_name}")

            scenario_dir = experiment_root / scenario_name

            # Use generated prompt + scenario base prompt
            combined_prompt = f"{generated_prompt}, {scenario_config['base_prompt']}"

            prompt_dir = scenario_dir / "prompt_0_generated"
            output_dir = str(prompt_dir)
            snapshot_info = []

            progress(base_progress + 0.1/total_scenarios, desc=f"Generating {scenario_name}...")

            # Generate main image
            input_img = Image.open(input_path).convert("RGB")
            if max(input_img.size) > MAX_RESOLUTION:
                ratio = MAX_RESOLUTION / max(input_img.size)
                new_size = tuple(int(dim * ratio // 16 * 16) for dim in input_img.size)
                input_img = input_img.resize(new_size, Image.Resampling.LANCZOS)

            input_img.save(prompt_dir / "input_image.png")

            generator = torch.Generator("cuda").manual_seed(42)

            with torch.no_grad():
                result = FLUX_PIPE(
                    prompt=combined_prompt,
                    image=input_img,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                    callback_on_step_end=decode_callback,
                    callback_on_step_end_tensor_inputs=["latents"]
                )

            result.images[0].save(prompt_dir / "final_output.png")

            progress(base_progress + 0.4/total_scenarios, desc=f"Creating evolution grid for {scenario_name}...")
            create_evolution_grid(prompt_dir)

            progress(base_progress + 0.6/total_scenarios, desc=f"Creating ablations for {scenario_name}...")
            create_word_attribution_ablation(FLUX_PIPE, input_path, combined_prompt, prompt_dir)

            # Save metadata
            metadata = {
                "prompt": combined_prompt,
                "scenario": scenario_name,
                "num_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "timestamp": datetime.now().isoformat()
            }

            with open(prompt_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            results[scenario_name] = str(prompt_dir / "final_output.png")

            torch.cuda.empty_cache()
            gc.collect()

        progress(1.0, desc="Complete!")

        # Return first scenario images for preview
        first_scenario = list(SCENARIOS.keys())[0]
        first_dir = experiment_root / first_scenario / "prompt_0_generated"

        final_img = str(first_dir / "final_output.png") if (first_dir / "final_output.png").exists() else None
        evolution_img = str(first_dir / "evolution_grid.png") if (first_dir / "evolution_grid.png").exists() else None
        attribution_img = str(first_dir / "word_attribution_complete.png") if (first_dir / "word_attribution_complete.png").exists() else None

        # Get first snapshot for timeline
        snapshot_dir = first_dir / "snapshots"
        first_snapshot = None
        if snapshot_dir.exists():
            snapshots = sorted(snapshot_dir.glob("step_*.png"))
            if snapshots:
                first_snapshot = str(snapshots[0])

        success_msg = f"""✅ **Generation Complete!**

**Experiment Path**: `{experiment_root}`

**Scenarios Generated**: {', '.join(SCENARIOS.keys())}

**Analysis Ready**: You can now use the analysis dashboard to review results and run LLM analysis.
"""

        return final_img, evolution_img, attribution_img, first_snapshot, success_msg, str(experiment_root)

    except Exception as e:
        import traceback
        error_msg = f"❌ **Generation Failed**\n\n```\n{traceback.format_exc()}\n```"
        return None, None, None, None, error_msg, None

# ===== BUILD GRADIO INTERFACE =====

with gr.Blocks(title="FLUX.1-Kontext Generator", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0; font-size: 2.5em;'>🎨 FLUX.1-Kontext Generator</h1>
        <p style='color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.2em;'>
            Upload Image → Generate Prompt → Create Variations → Analyze Results
        </p>
    </div>
    """)

    # State to hold experiment path
    experiment_path_state = gr.State()

    with gr.Tabs() as tabs:

        # TAB 1: Model Setup
        with gr.Tab("1️⃣ Setup", id=0):
            gr.Markdown("## 🚀 Initialize FLUX Model")
            gr.Markdown("Load the model once before generating. This may take 1-2 minutes.")

            load_btn = gr.Button("🔄 Load FLUX Model", variant="primary", size="lg")
            load_status = gr.Markdown("")

            load_btn.click(
                fn=load_flux_model,
                outputs=[load_status]
            )

        # TAB 2: Prompt Generation
        with gr.Tab("2️⃣ Generate Prompt", id=1):
            gr.Markdown("## 📸 Upload Image & Generate Prompt")

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Upload Input Image", type="numpy")
                    generate_prompt_btn = gr.Button("🤖 Generate Prompt with AI", variant="primary", size="lg")

                with gr.Column(scale=1):
                    generated_prompt = gr.Textbox(
                        label="Generated Prompt",
                        placeholder="Prompt will appear here...",
                        lines=5
                    )
                    safety_status = gr.Markdown("")

            generate_prompt_btn.click(
                fn=generate_prompt_from_image,
                inputs=[input_image],
                outputs=[generated_prompt, safety_status]
            )

        # TAB 3: Generation
        with gr.Tab("3️⃣ Generate Images", id=2):
            gr.Markdown("## 🎨 Generate All Scenarios")
            gr.Markdown("This will generate images for all scenarios (mug, t-shirt, gift bag) with word ablations and evolution grids.")

            generate_all_btn = gr.Button("🚀 Generate All Scenarios", variant="success", size="lg")
            generation_status = gr.Markdown("")

            gr.Markdown("### 📊 Preview (First Scenario)")

            with gr.Row():
                final_preview = gr.Image(label="Final Output", height=300)
                evolution_preview = gr.Image(label="Evolution Grid", height=300)

            with gr.Row():
                attribution_preview = gr.Image(label="Word Attribution", height=400)
                timeline_preview = gr.Image(label="First Snapshot", height=400)

            generate_all_btn.click(
                fn=run_flux_generation,
                inputs=[input_image, generated_prompt],
                outputs=[
                    final_preview,
                    evolution_preview,
                    attribution_preview,
                    timeline_preview,
                    generation_status,
                    experiment_path_state
                ]
            )

        # TAB 4: Next Steps
        with gr.Tab("4️⃣ Analysis", id=3):
            gr.Markdown("## 📊 Next: LLM Analysis")
            gr.Markdown("""
            After generation is complete:

            1. Copy the **Experiment Path** from the generation status
            2. Update `EXPERIMENT_ROOT` in the analysis dashboard code
            3. Run the analysis dashboard:
               ```bash
               python analysis_dashboard.py
               ```
            4. Browse results, run LLM analysis, and get prompt improvement suggestions

            ### Analysis Dashboard Features:
            - 🔍 Browse all generated scenarios
            - 🎬 Interactive timeline of diffusion steps
            - 🔬 Word attribution visualizations
            - 🤖 LLM-powered prompt analysis and suggestions
            """)

            experiment_path_display = gr.Textbox(
                label="Experiment Path (copy this)",
                interactive=False,
                value=experiment_path_state
            )

# ===== LAUNCH =====
if __name__ == "__main__":
    # Load model on startup
    print("🔄 Pre-loading FLUX model...")
    load_flux_model()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True
    )
