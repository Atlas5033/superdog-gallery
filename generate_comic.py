from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os
import random
from PIL import Image, ImageDraw, ImageFont

# Define moods and comic templates
moods_with_templates = {
    "Adventurous": [
        ["Smidge spots a mysterious cave in the forest.", "He steps inside, tail wagging with curiosity.",
         "Suddenly, he hears a distant echo and darts toward it.", "He finds an ancient bone buried in glowing crystals!"],
        ["Smidge gets a map in the mail marked with an X.", "He digs through his toy chest and grabs a compass.",
         "He hikes through the park, cape fluttering in the wind.", "He unearths a chest of squeaky toys under the old tree."],
        ["Smidge follows a trail of paw prints into the woods.", "He climbs a pile of logs to get a better view.",
         "A squirrel leads him on a chase through the trees.", "They end up sharing a snack at a hidden stream."],
        ["Smidge boards a cardboard rocket ship in the yard.", "He counts down and 'launches' into space.",
         "He 'lands' on the moon (aka sandbox) to explore.", "He plants a flag made of chew toys and barks with pride!"],
        ["Smidge finds a pair of goggles and pretends he's a pilot.", "He zooms down the hill in a wagon-turned-plane.",
         "Wind in his fur, he imagines he's flying over oceans.", "He lands in a pile of leaves and salutes the sky."],
        ["Smidge chases a feather into a neighbor’s backyard.", "It floats over a garden gnome and into a bush.",
         "He digs under the fence and emerges in a new world.", "He becomes king of the garden realm!"]
    ],
    "Sleepy": [
        ["Smidge yawns under a sunbeam on the couch.", "He drags his favorite blanket into a fluffy pile.",
         "A butterfly lands softly on his nose while he naps.", "He snoozes happily, dreaming of chasing squirrels."],
        ["Smidge climbs into a laundry basket full of warm towels.", "He wiggles into a cozy spot like a cinnamon roll.",
         "He dozes off as the dryer hums nearby.", "His dreams are filled with bouncing tennis balls."],
        ["Smidge finds a patch of grass in the yard and flops down.", "Clouds drift overhead as he slowly blinks.",
         "A leaf lands on his belly and he doesn't even flinch.", "He dozes peacefully, snoring gently."],
        ["Smidge curls up in a bean bag chair with a book beside him.", "He doesn’t read, of course — just smells the pages.",
         "The smell of paper and peanut butter relaxes him.", "He naps with the book like a bedtime buddy."],
        ["Smidge hides under the bed with a stuffed duck.", "The world outside is too loud today.",
         "He tucks his nose under his paws and closes his eyes.", "He dreams of a quiet island with endless naps."],
        ["Smidge is tucked into a child’s bed with a tiny pillow.", "The stars on the ceiling twinkle in the nightlight.",
         "He wiggles once to find the perfect position.", "Then drifts into dreamland surrounded by toys."]
    ],
    "Brave": [
        ["Smidge sees a cat stuck in a tree.", "He races over with his tiny cape flapping.",
         "He stacks pillows to climb up the tree base.", "The cat hops down — thanks to Smidge the Hero!"],
        ["Smidge hears thunder and hides behind the curtain.", "But then he remembers his superhero badge.",
         "He growls at the thunder and puffs up his chest.", "It still rains — but Smidge stands tall."],
        ["Smidge finds a frog trapped in a flower pot.", "He sniffs and paws at the edge to tip it over.",
         "The frog hops out with a thankful ribbit.", "Smidge watches it leap away, tail wagging proudly."],
        ["A balloon gets loose and floats toward a baby.", "Smidge chases it across the park, barking.",
         "He leaps and catches the string mid-air!", "The baby claps as Smidge returns the balloon."],
        ["A skateboard zooms toward an open gate.", "Smidge dashes across the yard to intercept it.",
         "He jumps on it and rolls it safely to a stop.", "His ears flap like a true stunt dog."],
        ["Smidge hears a whimper under the porch.", "He crawls inside and finds a scared kitten.",
         "He leads the kitten out with gentle barks.", "The kitten follows him like he’s the boss."]
    ],
    "Curious": [
        ["Smidge watches a snail crawl slowly across the sidewalk.", "He sniffs it once… then again.",
         "He gently boops it with his nose.", "He decides snails are friends — slow friends."],
        ["Smidge hears a beep from the microwave.", "He trots over and stares at the blinking numbers.",
         "His head tilts one way… then the other.", "He concludes it must be magic."],
        ["Smidge digs through a pile of socks.", "He finds one that smells like bacon.",
         "He parades it around the house triumphantly.", "Later, he hides it in his secret sock cave."],
        ["Smidge stares at his reflection in the mirror.", "He barks once — then two more times.",
         "Eventually, he brings the mirror a treat.", "Because friends deserve snacks."],
        ["Smidge follows a hummingbird through the yard.", "He tries to jump and follow its path.",
         "He spins in circles watching it dance.", "Eventually, he sits and just watches in awe."],
        ["Smidge finds a squeaky toy under the couch.", "He wiggles in with only his back legs sticking out.",
         "Ten minutes later, he emerges victorious.", "The toy squeaks in celebration."]
    ],
    "Joyful": [
        ["Smidge runs in circles with a new tennis ball.", "He tosses it in the air and catches it.",
         "He invites a bird to play (it declines).", "He rolls in the grass, just because."],
        ["Smidge sees his favorite human come home.", "His tail turns into a blur of joy.",
         "He leaps into their arms with a happy bark.", "They spin together in a happy reunion."],
        ["Smidge hears the ice cream truck jingle.", "He races toward it with his leash in his mouth.",
         "The driver gives him a peanut butter cone.", "He licks it with pure bliss under the summer sun."],
        ["Smidge jumps into a kiddie pool full of rubber ducks.", "He splashes like a cannonball hero.",
         "He chases bubbles through the air.", "Joy explodes with every wag."],
        ["Smidge watches fireworks with dog-safe headphones.", "Each burst makes his eyes sparkle.",
         "He does a happy dance between pops.", "The night ends with belly rubs and treats."],
        ["Smidge gets a surprise birthday party!", "All his neighborhood dog friends show up.",
         "There’s cake, toys, and barking laughter.", "He wears a party hat and grins for the camera."]
    ],
    "Gloomy": [
        ["Smidge sits by the window on a rainy day.", "His toy is soggy and the park is closed.",
         "He sighs and lays down with his chin on the floor.", "But a warm blanket soon makes it better."],
        ["Smidge loses his favorite squeaky toy.", "He searches under couches and behind chairs.",
         "No luck — it’s gone.", "He cuddles a backup duck but it’s not the same."],
        ["Smidge is left home alone while everyone goes out.", "He stares at the door, ears down.",
         "He howls softly once.", "Later, he finds a note that says 'We missed you!'"],
        ["Smidge sees his reflection and feels small today.", "His cape feels too heavy.",
         "He hides under the bed for a while.", "But later he peeks out — ready to try again."],
        ["Smidge hears thunder and burrows under the covers.", "The world feels too loud today.",
         "He whimpers once, then curls tighter.", "Eventually, the storm passes."],
        ["Smidge’s ice cream melts before he can eat it.", "It drips on his paw.",
         "He stares at it like it betrayed him.", "A friend brings him a new cone."]
    ]
}

# Select today's mood and story
mood_keys = sorted(moods_with_templates.keys())
today = datetime.utcnow().date()
mood_index = today.toordinal() % len(mood_keys)
daily_mood = mood_keys[mood_index]
frame_prompts = random.choice(moods_with_templates[daily_mood])

# Load AI model
print("🔄 Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    use_auth_token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Model loaded")

os.makedirs("images", exist_ok=True)

# Generate 4 image frames
frame_paths = []
for i, prompt in enumerate(frame_prompts, start=1):
    print(f"🎨 Generating frame {i}: {prompt}")
    full_prompt = f"A clean cartoon illustration of a dachshund superhero named Smidge, no text, no captions. {prompt}"
    image = pipe(full_prompt, num_inference_steps=5, guidance_scale=1.5).images[0]
    path = f"images/frame-{i}.png"
    image.save(path)
    frame_paths.append(path)
    print(f"✅ Saved {path}")

# Combine into 2x2 grid with captions

def wrap_text(text, font, max_width):
    lines = []
    words = text.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        if font.getlength(test_line) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

try:
    frames = [Image.open(fp) for fp in frame_paths]
    width, height = frames[0].size
    caption_height = 120
    padded_height = height + caption_height
    comic_strip = Image.new("RGB", (width * 2, padded_height * 2), color=(255, 255, 255))

    font_path = os.path.join(os.path.dirname(__file__), "Arial.ttf")
    font = ImageFont.truetype(font_path, 25)

    draw = ImageDraw.Draw(comic_strip)

    for idx, (img, caption) in enumerate(zip(frames, frame_prompts)):
        row = idx // 2
        col = idx % 2
        x = col * width
        y = row * padded_height

        comic_strip.paste(img, (x, y))
        draw.rectangle([x, y + height, x + width, y + padded_height], fill="white")
        wrapped = wrap_text(caption, font, width - 20)  # wrap to panel width
    for line_num, line in enumerate(wrapped):
        line_y = y + height + 10 + (line_num * font.size)
        draw.text((x + 10, line_y), line, fill="black", font=font)


    comic_strip.save("images/comic-strip.png")
    print("✅ Comic strip with captions saved")

except Exception as e:
    print("❌ Failed to create captioned comic strip:", e)

# Save mood label
with open("images/mood.txt", "w") as f:
    f.write(daily_mood)
print(f"📄 Mood saved: {daily_mood}")
