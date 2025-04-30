from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os
import random
from PIL import Image, ImageDraw, ImageFont

# Define moods and templates (trimmed example — you can paste the full list)
moods_with_templates = {
    "Adventurous": [
        ["Smidge spots a mysterious cave.", "He steps inside, tail wagging.", "He hears an echo and darts toward it.", "He finds an ancient bone in crystals!"]
    ],
    "Sleepy": [
        ["Smidge yawns under a sunbeam.", "He drags his blanket into a pile.", "A butterfly lands on his nose.", "He snoozes happily, dreaming of squirrels."]
    ]
    # Add other moods here...
}

# Select mood for today
mood_keys = sorted(moods_with_templates.keys())
today = datetime.utcnow().date()
mood_index = today.toordinal() % len(mood_keys)
daily_mood = mood_keys[mood_index]
frame_prompts = random.choice(moods_with_templates[daily_mood])

# Load model
print("🔄 Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    use_auth_token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Model loaded")

os.makedirs("images", exist_ok=True)

# Generate 4 comic panels
frame_paths = []
for i, prompt in enumerate(frame_prompts, start=1):
    print(f"🎨 Generating frame {i}: {prompt}")
    image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    path = f"images/frame-{i}.png"
    image.save(path)
    frame_paths.append(path)
    print(f"✅ Saved {path}")

# Combine into 2x2 comic
try:
    frames = [Image.open(fp) for fp in frame_paths]
    width, height = frames[0].size
    comic_strip = Image.new("RGB", (width * 2, height * 2), color=(255, 255, 255))
    comic_strip.paste(frames[0], (0, 0))
    comic_strip.paste(frames[1], (width, 0))
    comic_strip.paste(frames[2], (0, height))
    comic_strip.paste(frames[3], (width, height))

    # Add title
    draw = ImageDraw.Draw(comic_strip)
    text = f"Today's Mood: {daily_mood}"
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.rectangle([10, 10, 500, 70], fill="white")
    draw.text((20, 20), text, fill="black", font=font)

    comic_strip.save("images/comic-strip.png")
    print("✅ Comic strip saved")

except Exception as e:
    print("❌ Failed to create comic strip:", e)

# Save mood label
with open("images/mood.txt", "w") as f:
    f.write(daily_mood)
print(f"📄 Mood saved: {daily_mood}")
