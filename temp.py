"""
FLUX.1-Kontext Prompt Analysis Experiment Runner
Generates cross-attention style visualizations for prompt analysis
"""

import torch
from diffusers import FluxKontextPipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc
from collections import defaultdict
import json
from datetime import datetime
import time

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

INPUT_IMAGE = "holistic.png"
OUTPUT_ROOT = "flux_experiments"
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
MAX_RESOLUTION = 768

# Words to skip in ablation (common/filler words)
SKIP_WORDS = {'a', 'an', 'the', 'to', 'with', 'and', 'or', 'for', 'of', 'in', 'into'}

# ===== LATENT UNPACKING =====
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

# ===== GENERATION WITH TRACKING =====
snapshot_info = []
output_dir = None

def decode_callback(pipe_obj, step_index, timestep, callback_kwargs):
    """Decode and save intermediate latents"""
    global output_dir, snapshot_info
    
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
            
            del unpacked_latents, image_tensor, image, decoded
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  ⚠️ Failed at step {step_index}: {e}")
    
    return callback_kwargs

def generate_with_analysis(pipe, image_path, prompt, output_path):
    """Generate image with full analysis suite"""
    global output_dir, snapshot_info
    output_dir = str(output_path)
    snapshot_info = []
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and resize input
    input_image = Image.open(image_path).convert("RGB")
    if max(input_image.size) > MAX_RESOLUTION:
        ratio = MAX_RESOLUTION / max(input_image.size)
        new_size = tuple(int(dim * ratio // 16 * 16) for dim in input_image.size)
        input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
    
    input_image.save(Path(output_dir) / "input_image.png")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"  🎨 Generating: '{prompt[:60]}...'")
    start_time = time.time()
    
    generator = torch.Generator("cuda").manual_seed(42)
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            callback_on_step_end=decode_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        )
    
    generation_time = time.time() - start_time
    
    result.images[0].save(Path(output_dir) / "final_output.png")
    
    # Save metadata
    metadata = {
        "prompt": prompt,
        "num_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "generation_time_seconds": generation_time,
        "snapshots_saved": len(snapshot_info),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ Generated in {generation_time:.1f}s")
    
    return result.images[0], input_image, metadata

# ===== SMART WORD SELECTION =====
def select_important_words(prompt, max_words=4):
    """Select the most important/descriptive words for ablation"""
    words = prompt.split()
    
    # Filter out common words
    important_words = [w for w in words if w.lower() not in SKIP_WORDS]
    
    # If still too many, prioritize nouns/adjectives (heuristic: longer words)
    if len(important_words) > max_words:
        important_words = sorted(important_words, key=len, reverse=True)[:max_words]
    
    print(f"    Selected words for ablation: {important_words}")
    return important_words

# ===== VISUALIZATIONS =====
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
    print(f"    ✓ Saved evolution_grid.png")

def create_word_attribution_ablation(pipe, image_path, prompt, output_dir):
    """
    Generate complete word attribution showing:
    - Original images
    - Ablated images (word removed)
    - Difference heatmaps
    """
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
    
    # Smart word selection
    important_words = select_important_words(prompt, max_words=4)
    
    word_data = {}  # Store both ablated images and heatmaps
    
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
    
    # Create 3-ROW visualization
    n_words = len(word_data)
    fig, axes = plt.subplots(3, n_words, figsize=(4*n_words, 12))
    if n_words == 1:
        axes = axes.reshape(3, 1)
    
    for i, (word, data) in enumerate(word_data.items()):
        # ROW 1: Baseline with full prompt
        axes[0, i].imshow(baseline_img.astype(np.uint8))
        axes[0, i].set_title(f'WITH "{word}"', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')
        
        # ROW 2: Ablated (word removed)
        axes[1, i].imshow(data['ablated_image'].astype(np.uint8))
        axes[1, i].set_title(f'WITHOUT "{word}"', fontsize=11, fontweight='bold', color='red')
        axes[1, i].axis('off')
        
        # ROW 3: Difference heatmap
        im = axes[2, i].imshow(data['heatmap'], cmap='hot', interpolation='bilinear')
        axes[2, i].set_title(f'Difference Map', fontsize=10)
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)
    
    plt.suptitle(f'Word Attribution Analysis\nPrompt: "{prompt}"', 
                 fontsize=13, y=0.99, fontweight='bold')
    
    # Add row labels on the left
    fig.text(0.02, 0.75, 'Baseline\n(Full Prompt)', 
             ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.50, 'Ablated\n(Word Removed)', 
             ha='center', va='center', fontsize=12, fontweight='bold', rotation=90, color='red')
    fig.text(0.02, 0.25, 'Change\nHeatmap', 
             ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    plt.savefig(Path(output_dir) / "word_attribution_complete.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved word_attribution_complete.png")
    
    # ALSO save individual ablated images for inspection
    for word, data in word_data.items():
        ablated_pil = Image.fromarray(data['ablated_image'].astype(np.uint8))
        ablated_pil.save(Path(output_dir) / f"ablated_without_{word}.png")
    
    print(f"    ✓ Saved {len(word_data)} individual ablated images")
    
    return word_data

def create_timestep_word_evolution(output_dir, tracked_word):
    """
    Create timestep-based attention evolution (BOTTOM ROW of reference image)
    Uses the snapshots already generated during main inference
    """
    snapshot_dir = Path(output_dir) / "snapshots"
    if not snapshot_dir.exists():
        print("    ⚠️ No snapshots found for timestep evolution")
        return
    
    snapshot_files = sorted(snapshot_dir.glob("step_*.png"))
    
    if len(snapshot_files) == 0:
        return
    
    # Create bottom row visualization
    n_steps = len(snapshot_files)
    fig, axes = plt.subplots(1, n_steps, figsize=(3*n_steps, 3))
    if n_steps == 1:
        axes = [axes]
    
    for i, snap_file in enumerate(snapshot_files):
        img = Image.open(snap_file)
        axes[i].imshow(img)
        
        # Extract step number from filename
        step_num = snap_file.stem.split('_')[1]
        axes[i].set_title(f"t={step_num}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f'Cross-Attention Maps for Individual Timestamps\n"{tracked_word}"', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"timestep_evolution_{tracked_word}.png", 
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved timestep_evolution_{tracked_word}.png")

# ===== MAIN EXPERIMENT RUNNER =====
def run_full_experiment():
    """Run complete experiment across all scenarios"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = Path(OUTPUT_ROOT) / f"run_{timestamp}"
    experiment_root.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config = {
        "scenarios": SCENARIOS,
        "input_image": INPUT_IMAGE,
        "inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "timestamp": timestamp
    }
    
    with open(experiment_root / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load model once (reuse for all generations)
    print("🔄 Loading FLUX.1-Kontext-dev model...")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    print("✅ Model loaded!\n")
    
    results_summary = {}
    
    # Iterate through scenarios
    for scenario_name, scenario_config in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        scenario_dir = experiment_root / scenario_name
        scenario_dir.mkdir(exist_ok=True)
        
        scenario_results = {}
        
        # Test all prompts (base + variants)
        all_prompts = [scenario_config["base_prompt"]] + scenario_config["variants"]
        
        for prompt_idx, prompt in enumerate(all_prompts):
            prompt_name = f"prompt_{prompt_idx}_base" if prompt_idx == 0 else f"prompt_{prompt_idx}_variant{prompt_idx}"
            
            print(f"\n  📝 Prompt {prompt_idx + 1}/{len(all_prompts)}")
            
            # Output directory
            output_path = scenario_dir / prompt_name
            
            # Generate with analysis
            final_img, input_img, metadata = generate_with_analysis(
                pipe, INPUT_IMAGE, prompt, output_path
            )
            
            # Create evolution grid
            print("    Creating evolution grid...")
            create_evolution_grid(output_path)
            
            # Create word attribution map (smart ablation)
            print("    Creating word attribution map...")
            word_heatmaps = create_word_attribution_ablation(
                pipe, INPUT_IMAGE, prompt, output_path
            )
            
            # Create timestep evolution for first important word
            important_words = select_important_words(prompt, max_words=1)
            if important_words:
                print(f"    Creating timestep evolution for '{important_words[0]}'...")
                create_timestep_word_evolution(output_path, important_words[0])
            
            scenario_results[prompt_name] = {
                "prompt": prompt,
                "output_dir": str(output_path),
                "generation_time": metadata["generation_time_seconds"],
                "final_image": str(output_path / "final_output.png")
            }
            
            torch.cuda.empty_cache()
            gc.collect()
        
        results_summary[scenario_name] = scenario_results
    
    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    
    # Save results summary
    with open(experiment_root / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ EXPERIMENT COMPLETE!")
    print(f"📁 Results saved to: {experiment_root}")
    print(f"📊 Total prompts tested: {sum(len(s) for s in results_summary.values())}")
    print(f"{'='*60}")
    
    return experiment_root, results_summary

# ===== RUN DIRECTLY IN NOTEBOOK =====
experiment_path, summary = run_full_experiment()
print(f"\n🎉 Ready for LLM analysis!")
print(f"   Experiment path: {experiment_path}")