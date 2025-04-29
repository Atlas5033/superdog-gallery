from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os

# Load the AI model
print("🔄 Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    use_auth_token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Model loaded")

# Create output folder
os.makedirs("images", exist_ok=True)

# Generate Superdog image for current hour
hour = datetime.utcnow().hour
prompt = "A cartoon dachshund in a superhero costume flying through the sky, comic book style"
image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save(f"images/superdog-hour-{hour}.png")
image.save("images/latest.png")
print(f"✅ Superdog image saved for hour {hour}")
