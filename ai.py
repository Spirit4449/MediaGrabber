from google import genai
from dotenv import load_dotenv
import os

# --- setup ---
load_dotenv()
client = genai.Client()

# folder of images
FOLDER = "images"  # change this folder name as needed

PROMPT = """
Analyze this image and reply only in valid JSON:
{
  "tags": ["tag1", ..., "tag10"],
  "inspiration": true | false
}

Rules:
- Create exactly 10 concise tags describing design aspects (style, color, layout, subject, mood, etc.).
- "inspiration": true only if the image shows strong, modern, and creative design that a content creator could learn from or be visually inspired by — e.g. thoughtful layout, balanced spacing, appealing typography, clean hierarchy, professional color use, or original concept.
- "inspiration": false if the design looks generic, outdated, templated, over-decorated, overly saturated, inconsistent in color, text-heavy, poorly balanced, or lacks originality.
- Event or invitation flyers may still count as inspiration, but only if they show deliberate and tasteful design quality — not just festive or decorative templates.
- Always mark photographs, plain screenshots, selfies, or non-design content as false.
- Return valid JSON only, no markdown or explanations.
"""

# get list of image files
images = [f for f in os.listdir(FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))]
images.sort()

print(f"Found {len(images)} images in '{FOLDER}'.\nPress ENTER to analyze next or 'q' to quit.\n")

for idx, filename in enumerate(images, start=1):
    image_path = os.path.join(FOLDER, filename)
    choice = input(f"[{idx}/{len(images)}] {filename} → Press Enter to analyze (q to quit): ")
    if choice.lower() == "q":
        break

    try:
        # upload and analyze
        my_file = client.files.upload(file=image_path)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[my_file, PROMPT]
        )

        print("\n🔹 Result:\n", response.text, "\n")
        client.files.delete(name=my_file.name)  # cleanup uploaded file

    except Exception as e:
        print(f"❌ Error on {filename} →", e, "\n")
