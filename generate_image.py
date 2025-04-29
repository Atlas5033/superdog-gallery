from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os
import random

# Define moods and comic templates
moods_with_templates = {
    "Adventurous": [
        ["Smidge spots a mysterious cave in the forest.", "He steps inside, tail wagging with curiosity.", "Suddenly, he hears a distant echo and darts toward it.", "He finds an ancient bone buried in glowing crystals!"],
        ["Smidge gets a map in the mail marked with an X.", "He digs through his toy chest and grabs a compass.", "He hikes through the park, cape fluttering in the wind.", "He unearths a chest of squeaky toys under the old tree."]
    ],
    "Sleepy": [
        ["Smidge yawns under a sunbeam on the couch.", "He drags his favorite blanket into a fluffy pile.", "A butterfly lands softly on his nose while he naps.", "He snoozes happily, dreaming of chasing squirrels."]
    ],
    "Brave": [
        ["Smidge sees a cat stuck in a tree.", "He races over with his tiny cape flapping.", "He stacks pillows to climb up the tree base.", "The cat hops down — thanks to Smidge the Hero!"]
    ]
    # You can add more moods here
}

# Determine today's mood
mood_keys = sorted(moods_with_templates.keys())
today = datetime.utcnow().date()
mood_index = today.toordinal() % len(mood_keys)
daily_mood = mood_keys[mood_index]
frame_prompts = random.choice(moods_with_templates[daily_mood])

# Load the model
print("🔄 Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    use_auth_token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.float32
).to("cpu")
print("✅ Model loaded")

# Create images folder if needed
os.makedirs("images", exist_ok=True)

# Generate hourly hero image
hour = datetime.utcnow().hour
hourly_prompt = "A cartoon dachshund in a superhero costume flying through the sky, comic book style"
print(f"🎨 Generating Superdog of the Hour ({hour})...")
image = pipe(hourly_prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save(f"images/superdog-hour-{hour}.png")
image.save("images/latest.png")
print("✅ Superdog saved")

# Generate 4-panel comic
for i, prompt in enumerate(frame_prompts, start=1):
    print(f"🎨 Generating frame {i}: {prompt}")
    comic_image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    comic_image.save(f"images/frame-{i}.png")
    print(f"✅ Saved frame-{i}.png")

# Save mood for HTML display
with open("images/mood.txt", "w") as f:
    f.write(daily_mood)
print(f"📄 Mood saved: {daily_mood}")
