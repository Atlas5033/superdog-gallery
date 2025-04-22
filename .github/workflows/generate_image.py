from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os

print("Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    use_auth_token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.float32  # Use float32 for CPU
).to("cpu")

# Use the current hour
hour = datetime.utcnow().hour
filename = f"images/superdog-hour-{hour}.png"

prompt = "A cartoon dachshund in a superhero costume flying through the sky, comic book style"

print("Generating image...")
image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

print("Saving image...")
os.makedirs("images", exist_ok=True)
image.save(filename)
print(f"✅ Saved image as: {filename}")
