from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    use_auth_token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.float16,
).to("cuda" if torch.cuda.is_available() else "cpu")

hour = datetime.utcnow().hour
filename = f"images/superdog-hour-{hour}.png"

prompt = "A cartoon dachshund dog in a superhero costume flying through the sky, colorful, comic book style"

image = pipe(prompt).images[0]

os.makedirs("images", exist_ok=True)
image.save(filename)
