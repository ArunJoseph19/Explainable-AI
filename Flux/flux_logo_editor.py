#!/usr/bin/env python3
"""
FLUX.1 Kontext Logo Editing with Prompt-to-Prompt Analysis
Quick Start Script

Usage:
    python flux_logo_editor.py --logo company_logo.png --output ./results
"""

import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from datetime import datetime
import argparse


class LogoEditorPipeline:
    """Complete pipeline for iterative logo editing and analysis"""
    
    def __init__(self, output_dir="./logo_edits", device="cuda"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        
        print("🚀 Loading FLUX.1 Kontext...")
        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16
        ).to(device)
        print("✓ Model loaded successfully")
        
        self.edit_history = []
    
    def edit_image(
        self,
        image_path: str,
        edit_instruction: str,
        edit_type: str = "refinement",
        strength: float = 0.8,
        num_steps: int = 30,
        seed: int = 42
    ) -> Image.Image:
        """
        Edit a logo/image with a text instruction
        
        Args:
            image_path: Path to input image
            edit_instruction: Text describing the edit (e.g., "add christmas vibe")
            edit_type: 'refinement' (add) or 'word_swap' (replace)
            strength: How much to modify (0-1, higher = more change)
            num_steps: Denoising steps (higher = better quality, slower)
            seed: Random seed for reproducibility
        
        Returns:
            Edited PIL Image
        """
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 1024))
        
        print(f"\n📝 Edit: {edit_instruction}")
        print(f"   Type: {edit_type} | Strength: {strength} | Steps: {num_steps}")
        
        # Generate with FLUX
        with torch.no_grad():
            output = self.pipeline(
                prompt=edit_instruction,
                image=image,
                num_inference_steps=num_steps,
                strength=strength,
                generator=torch.Generator(self.device).manual_seed(seed)
            )
        
        edited_image = output.images[0]
        
        # Log edit
        self.edit_history.append({
            'timestamp': datetime.now().isoformat(),
            'instruction': edit_instruction,
            'type': edit_type,
            'strength': strength,
            'num_steps': num_steps,
            'seed': seed
        })
        
        return edited_image
    
    def batch_edit(self, initial_logo: str, edits: list) -> list:
        """
        Apply multiple edits sequentially to a logo
        
        Args:
            initial_logo: Path to company logo
            edits: List of dicts with keys:
                - 'instruction' (str): Edit description
                - 'type' (str): 'refinement' or 'word_swap'
                - 'strength' (float): Optional, default 0.8
                - 'steps' (int): Optional, default 30
        
        Returns:
            List of edited images
        """
        
        print(f"\n{'='*70}")
        print(f"📦 BATCH EDITING: {len(edits)} edits")
        print(f"{'='*70}")
        
        images = [Image.open(initial_logo).convert("RGB").resize((1024, 1024))]
        
        for idx, edit_spec in enumerate(edits):
            print(f"\n[{idx+1}/{len(edits)}] ", end="")
            
            # Use previous output as input
            current = images[-1]
            
            # Save temp image
            temp_path = self.output_dir / f"temp_step_{idx}.png"
            current.save(temp_path)
            
            # Edit
            edited = self.edit_image(
                str(temp_path),
                edit_spec['instruction'],
                edit_type=edit_spec.get('type', 'refinement'),
                strength=edit_spec.get('strength', 0.8),
                num_steps=edit_spec.get('steps', 30),
                seed=idx  # Use index as seed for consistency
            )
            
            images.append(edited)
            
            # Save output
            output_path = self.output_dir / f"edit_{idx:02d}_{edit_spec['type']}.png"
            edited.save(output_path)
            print(f"✓ Saved: {output_path.name}")
        
        return images
    
    def compare_images(self, img1: Image.Image, img2: Image.Image, title: str = ""):
        """
        Compare two images side-by-side with difference map
        """
        
        arr1 = np.array(img1) / 255.0
        arr2 = np.array(img2) / 255.0
        diff = np.abs(arr1 - arr2).mean(axis=2)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        axes[0].imshow(img1)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title("Edited")
        axes[1].axis('off')
        
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title("Pixel Differences")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], label='Max Change')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Save
        output_file = self.output_dir / f"comparison_{title.replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Compute statistics
        change_pct = (diff > 0.1).sum() / diff.size * 100
        print(f"   └─ {change_pct:.1f}% of image changed")
        
        return diff
    
    def create_comparison_grid(self, images: list) -> Image.Image:
        """
        Create a grid showing all edits
        """
        
        n = len(images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Step {idx}")
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for idx in range(n, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle("Logo Editing Journey", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / "edit_grid.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved grid: {output_file.name}")
    
    def save_metadata(self):
        """Save edit history as JSON"""
        
        metadata_file = self.output_dir / "edit_history.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.edit_history, f, indent=2)
        
        print(f"✓ Saved metadata: {metadata_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.1 Kontext Logo Editor with Analysis"
    )
    parser.add_argument('--logo', required=True, help='Path to company logo')
    parser.add_argument('--output', default='./logo_edits', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    editor = LogoEditorPipeline(output_dir=args.output, device=args.device)
    
    # Define edits for company logo
    edits = [
        {
            'instruction': 'add a festive Christmas theme with snow and holiday lights',
            'type': 'refinement',
            'strength': 0.8,
            'steps': 30
        },
        {
            'instruction': 'add a red Santa hat on top',
            'type': 'refinement',
            'strength': 0.7,
            'steps': 30
        },
        {
            'instruction': 'add a cozy winter scarf around it',
            'type': 'refinement',
            'strength': 0.75,
            'steps': 30
        }
    ]
    
    # Run batch editing
    edited_images = editor.batch_edit(args.logo, edits)
    
    # Analyze transformations
    print(f"\n{'='*70}")
    print("📊 ANALYZING TRANSFORMATIONS")
    print(f"{'='*70}")
    
    for idx in range(1, len(edited_images)):
        print(f"\n[Comparison {idx}] Step {idx-1} → Step {idx}")
        editor.compare_images(
            edited_images[idx-1],
            edited_images[idx],
            title=f"Edit_{idx}"
        )
    
    # Create grid
    editor.create_comparison_grid(edited_images)
    
    # Save metadata
    editor.save_metadata()
    
    print(f"\n{'='*70}")
    print(f"✅ Complete! Results saved to: {editor.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
