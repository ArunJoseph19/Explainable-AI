from flask import Flask, request, jsonify
import torch
from diffusers import FluxPipeline
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

print("Loading FLUX.1-Kontext-dev...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    use_safetensors=True  # Add this
)
pipe.to("cuda")
print("✅ Model loaded!")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        image_b64 = data['image']
        prompt = data['prompt']
        strength = data.get('strength', 0.8)
        steps = data.get('steps', 8)
        seed = data.get('seed', 42)
        
        # Decode input
        image_bytes = base64.b64decode(image_b64)
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Store timesteps
        timesteps_output = []
        
        def save_timestep(pipe, step_index, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            with torch.no_grad():
                image = pipe.vae.decode(
                    latents / pipe.vae.config.scaling_factor, 
                    return_dict=False
                )[0]
                image = pipe.image_processor.postprocess(image, output_type="pil")[0]
            
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            timesteps_output.append({"step": step_index, "image": img_b64})
            return callback_kwargs
        
        # Generate
        generator = torch.Generator("cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            num_inference_steps=steps,
            generator=generator,
            callback_on_step_end=save_timestep,
            callback_on_step_end_tensor_inputs=["latents"]
        )
        
        # Encode final
        buffer = BytesIO()
        result.images[0].save(buffer, format="PNG")
        final_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'final_image': final_b64,
            'timesteps': timesteps_output,
            'num_steps': len(timesteps_output)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'FLUX.1-Kontext-dev'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
