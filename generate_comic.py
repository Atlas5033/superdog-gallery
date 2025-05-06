from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os
import random
from PIL import Image, ImageDraw, ImageFont

# Define moods and comic templates
moods_with_templates = {
    "Adventurous": [
        ["Smidge spots a mysterious cave in the forest.", "She steps inside, tail wagging with curiosity.",
         "Suddenly, she hears a distant echo and darts toward it.", "She finds an ancient bone buried in glowing crystals!"],
        ["Smidge gets a map in the mail marked with an X.", "She digs through her toy chest and grabs a compass.",
         "She hikes through the park, cape fluttering in the wind.", "She unearths a chest of squeaky toys under the old tree."],
        ["Smidge follows a trail of paw prints into the woods.", "She climbs a pile of logs to get a better view.",
         "A squirrel leads her on a chase through the trees.", "They end up sharing a snack at a hidden stream."],
        ["Smidge boards a cardboard rocket ship in the yard.", "She counts down and 'launches' into space.",
         "She 'lands' on the moon (aka sandbox) to explore.", "She plants a flag made of chew toys and barks with pride!"],
        ["Smidge finds a pair of goggles and pretends she's a pilot.", "She zooms down the hill in a wagon-turned-plane.",
         "Wind in her fur, she imagines she's flying over oceans.", "She lands in a pile of leaves and salutes the sky."],
        ["Smidge chases a feather into a neighbor’s backyard.", "It floats over a garden gnome and into a bush.",
         "She digs under the fence and emerges in a new world.", "She becomes queen of the garden realm!"]
    ],
    "Sleepy": [
        ["Smidge yawns under a sunbeam on the couch.", "She drags her favorite blanket into a fluffy pile.",
         "A butterfly lands softly on her nose while she naps.", "She snoozes happily, dreaming of chasing squirrels."],
        ["Smidge climbs into a laundry basket full of warm towels.", "She wiggles into a cozy spot like a cinnamon roll.",
         "She dozes off as the dryer hums nearby.", "Her dreams are filled with bouncing tennis balls."],
        ["Smidge finds a patch of grass in the yard and flops down.", "Clouds drift overhead as she slowly blinks.",
         "A leaf lands on her belly and she doesn't even flinch.", "She dozes peacefully, snoring gently."],
        ["Smidge curls up in a bean bag chair with a book beside her.", "She doesn’t read, of course — just smells the pages.",
         "The smell of paper and peanut butter relaxes her.", "She naps with the book like a bedtime buddy."],
        ["Smidge hides under the bed with a stuffed duck.", "The world outside is too loud today.",
         "She tucks her nose under her paws and closes her eyes.", "She dreams of a quiet island with endless naps."],
        ["Smidge is tucked into a child’s bed with a tiny pillow.", "The stars on the ceiling twinkle in the nightlight.",
         "She wiggles once to find the perfect position.", "Then drifts into dreamland surrounded by toys."]
    ],
    "Brave": [
        ["Smidge sees a cat stuck in a tree.", "She races over with her tiny cape flapping.",
         "She stacks pillows to climb up the tree base.", "The cat hops down — thanks to Smidge the Hero!"],
        ["Smidge hears thunder and hides behind the curtain.", "But then she remembers her superhero badge.",
         "She growls at the thunder and puffs up her chest.", "It still rains — but Smidge stands tall."],
        ["Smidge finds a frog trapped in a flower pot.", "She sniffs and paws at the edge to tip it over.",
         "The frog hops out with a thankful ribbit.", "Smidge watches it leap away, tail wagging proudly."],
        ["A balloon gets loose and floats toward a baby.", "Smidge chases it across the park, barking.",
         "She leaps and catches the string mid-air!", "The baby claps as Smidge returns the balloon."],
        ["A skateboard zooms toward an open gate.", "Smidge dashes across the yard to intercept it.",
         "She jumps on it and rolls it safely to a stop.", "Her ears flap like a true stunt dog."],
        ["Smidge hears a whimper under the porch.", "She crawls inside and finds a scared kitten.",
         "She leads the kitten out with gentle barks.", "The kitten follows her like she’s the boss."]
    ],
    "Curious": [
        ["Smidge watches a snail crawl slowly across the sidewalk.", "She sniffs it once… then again.",
         "She gently boops it with her nose.", "She decides snails are friends — slow friends."],
        ["Smidge hears a beep from the microwave.", "She trots over and stares at the blinking numbers.",
         "Her head tilts one way… then the other.", "She concludes it must be magic."],
        ["Smidge digs through a pile of socks.", "She finds one that smells like bacon.",
         "She parades it around the house triumphantly.", "Later, she hides it in her secret sock cave."],
        ["Smidge stares at her reflection in the mirror.", "She barks once — then two more times.",
         "Eventually, she brings the mirror a treat.", "Because friends deserve snacks."],
        ["Smidge follows a hummingbird through the yard.", "She tries to jump and follow its path.",
         "She spins in circles watching it dance.", "Eventually, she sits and just watches in awe."],
        ["Smidge finds a squeaky toy under the couch.", "She wiggles in with only her back legs sticking out.",
         "Ten minutes later, she emerges victorious.", "The toy squeaks in celebration."]
    ],
    "Joyful": [
        ["Smidge runs in circles with a new tennis ball.", "She tosses it in the air and catches it.",
         "She invites a bird to play (it declines).", "She rolls in the grass, just because."],
        ["Smidge sees her favorite human come home.", "Her tail turns into a blur of joy.",
         "She leaps into their arms with a happy bark.", "They spin together in a happy reunion."],
        ["Smidge hears the ice cream truck jingle.", "She races toward it with her leash in her mouth.",
         "The driver gives her a peanut butter cone.", "She licks it with pure bliss under the summer sun."],
        ["Smidge jumps into a kiddie pool full of rubber ducks.", "She splashes like a cannonball hero.",
         "She chases bubbles through the air.", "Joy explodes with every wag."],
        ["Smidge watches fireworks with dog-safe headphones.", "Each burst makes her eyes sparkle.",
         "She does a happy dance between pops.", "The night ends with belly rubs and treats."],
        ["Smidge gets a surprise birthday party!", "All her neighborhood dog friends show up.",
         "There’s cake, toys, and barking laughter.", "She wears a party hat and grins for the camera."]
    ],
    "Gloomy": [
        ["Smidge sits by the window on a rainy day.", "Her toy is soggy and the park is closed.",
         "She sighs and lays down with her chin on the floor.", "But a warm blanket soon makes it better."],
        ["Smidge loses her favorite squeaky toy.", "She searches under couches and behind chairs.",
         "No luck — it’s gone.", "She cuddles a backup duck but it’s not the same."],
        ["Smidge is left home alone while everyone goes out.", "She stares at the door, ears down.",
         "She howls softly once.", "Later, she finds a note that says 'We missed you!'"] ,
        ["Smidge sees her reflection and feels small today.", "Her cape feels too heavy.",
         "She hides under the bed for a while.", "But later she peeks out — ready to try again."],
        ["Smidge hears thunder and burrows under the covers.", "The world feels too loud today.",
         "She whimpers once, then curls tighter.", "Eventually, the storm passes."],
        ["Smidge’s ice cream melts before she can eat it.", "It drips on her paw.",
         "She stares at it like it betrayed her.", "A friend brings her a new cone."]
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
        wrapped = wrap_text(caption, font, width - 20)

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
