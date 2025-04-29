from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os

print("🔄 Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    use_auth_token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.float32
).to("cpu")
print("✅ Model loaded")

hour = datetime.utcnow().hour
filename = f"images/superdog-hour-{hour}.png"
prompt = "A cartoon dachshund in a superhero costume flying through the sky, comic book style"

print("🎨 Generating image...")
image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
print("✅ Image generated")

os.makedirs("images", exist_ok=True)
image.save(filename)
print(f"✅ Image saved as {filename}")

# Also save as latest.png for website display
image.save("images/latest.png")
print("✅ Also saved as images/latest.png")

from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime
import os
import random

# Define moods and multiple 4-panel templates per mood
moods_with_templates = {
    "Adventurous": [
        ["Smidge spots a mysterious cave in the forest.", "He steps inside, tail wagging with curiosity.", "Suddenly, he hears a distant echo and darts toward it.", "He finds an ancient bone buried in glowing crystals!"],
        ["Smidge gets a map in the mail marked with an X.", "He digs through his toy chest and grabs a compass.", "He hikes through the park, cape fluttering in the wind.", "He unearths a chest of squeaky toys under the old tree."],
        ["Smidge climbs aboard a toy rocket ship.", "He blasts off toward the stars with a bark.", "He floats in zero gravity chasing floating treats.", "He lands on a planet made entirely of tennis balls."],
        ["Smidge hears a whistle and sees a parade forming.", "He hops into the lineup beside jugglers and clowns.", "He dazzles the crowd with flips and spins.", "He bows as confetti falls all around him."],
        ["Smidge finds a glowing doorway in the backyard.", "He steps through and finds a jungle filled with animals.", "He makes friends with a baby tiger.", "They share snacks under a giant fern."]
    ],
    "Sleepy": [
        ["Smidge yawns under a sunbeam on the couch.", "He drags his favorite blanket into a fluffy pile.", "A butterfly lands softly on his nose while he naps.", "He snoozes happily, dreaming of chasing squirrels."],
        ["Smidge curls up in a laundry basket full of warm clothes.", "He rearranges the socks into a perfect pillow.", "He hears a noise but refuses to open his eyes.", "Still half-asleep, he smiles and stretches his little paws."],
        ["Smidge climbs into a hammock swinging in the breeze.", "He listens to birds chirping and slowly drifts off.", "He dreams of floating on a cloud made of marshmallows.", "He wakes up with one ear flopped over his eyes."],
        ["Smidge piles up his plush toys like a mountain.", "He scales the top and settles like a king.", "He lets out a sleepy sigh.", "A gentle snore escapes as the toys shift."],
        ["Smidge finds an open book on the floor.", "He lies on it and flips a page with his paw.", "Soon his eyes get heavy.", "He falls asleep dreaming of the story."]
    ],
    "Brave": [
        ["Smidge sees a cat stuck in a tree.", "He races over with his tiny cape flapping.", "He stacks pillows to climb up the tree base.", "The cat hops down — thanks to Smidge the Hero!"],
        ["Smidge hears thunder rumbling in the distance.", "He puts on his raincoat and heads into the storm.", "Lightning flashes, but he presses forward boldly.", "He saves a stranded toy from a puddle and smiles."],
        ["Smidge watches as balloons fly off a cart.", "He jumps high and snags the ribbon in his teeth.", "The children cheer and lift him up.", "He becomes honorary balloon captain."],
        ["Smidge sees a lost puppy in the park.", "He sniffs out the scent and follows the trail.", "He leads the pup back to a grateful family.", "They give him a medal made of treats."],
        ["Smidge finds a large box with a mystery sound.", "He carefully nudges the lid open.", "A bird flutters out and lands on his nose.", "He stands still, brave and calm."]
    ],
    "Playful": [
        ["Smidge finds a bouncing red ball in the yard.", "He pounces on it, flipping with excitement.", "The ball escapes! Smidge chases it across the lawn.", "He finally catches it and poses like a champ!"],
        ["Smidge invites a squirrel to a race.", "They dash through bushes and under park benches.", "Smidge leaps through a hoop with flair.", "Both collapse in laughter, tails wagging."],
        ["Smidge finds a garden hose spraying in the sun.", "He leaps into the spray, barking joyfully.", "He chases rainbows and tries to bite them.", "He ends soaked but smiling."],
        ["Smidge builds a fort from couch cushions.", "He drags his toys inside one by one.", "He peeks out like a knight on watch.", "Then rolls over giggling."],
        ["Smidge sees his reflection in a mirror.", "He barks and wags his tail.", "He tries to paw the other pup.", "Then realizes it’s just goofy him."]
    ],
    "Curious": [
        ["Smidge discovers a locked box in the backyard.", "He sniffs all around it and finds a tiny key.", "The key fits! He lifts the lid slowly.", "Inside is a collection of tiny dog-sized hats."],
        ["Smidge notices a buzzing sound in the flowerbed.", "He tiptoes closer, ears perked up.", "A bumblebee hovers nearby — not scary, just friendly.", "They circle each other curiously, then part ways."],
        ["Smidge watches a robot toy blink its lights.", "He noses it and it begins to roll.", "He chases it through the kitchen.", "Eventually, he hugs it like a friend."],
        ["Smidge spots an envelope under the door.", "He drags it open and sniffs.", "It's a scented invitation to a secret garden party!", "He tilts his head and winks at the camera."],
        ["Smidge sees shadows dancing on the wall.", "He tries to catch them with his paws.", "He looks behind him to find the light.", "He sits in the beam, becoming the shadow himself."]
    ],
    "Silly": [
        ["Smidge puts on oversized sunglasses.", "He struts around like a movie star.", "He bumps into a trash can — whoops!", "He pops up and bows like it was planned."],
        ["Smidge gets stuck in a toilet paper roll.", "He wiggles out with dramatic flair.", "He wraps the roll around himself like a toga.", "He howls in triumph."],
        ["Smidge tries to balance a cookie on his nose.", "He crosses his eyes concentrating.", "The cookie drops! He leaps and catches it midair.", "Victory pose with crumbs all over his face."],
        ["Smidge rides a toy skateboard down the hallway.", "He zooms past the cat in a blur.", "He hits a pillow ramp and lands in a pile of laundry.", "The cat claps."],
        ["Smidge chases his tail.", "Faster and faster until he falls over dizzy.", "He sits and laughs at himself.", "Then starts again."],
        ["Smidge wears socks on all four paws.", "He slides like a penguin across the floor.", "He bumps the door and grins.", "He’s proud of his stunt show."]
    ]
}

# Pick today's mood based on the date
mood_keys = sorted(moods_with_templates.keys())
today = datetime.utcnow().date()
mood_index = today.toordinal() % len(mood_keys)
daily_mood = mood_keys[mood_index]

# Pick a random 4-panel story from that mood
frame_prompts = random.choice(moods_with_templates[daily_mood])

# Load the model
print("🔄 Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    use_auth_token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.float32
).to("cpu")
print("✅ Model loaded")

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Generate 4-frame comic
for i, prompt in enumerate(frame_prompts, start=1):
    print(f"🎨 Generating frame {i}: {prompt}")
    image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    filename = f"images/frame-{i}.png"
    image.save(filename)
    print(f"✅ Saved {filename}")

# Save mood to text file
with open("images/mood.txt", "w") as f:
    f.write(daily_mood)
print(f"📄 Mood saved: {daily_mood}")

