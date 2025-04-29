import os
import json
import whisper

model = whisper.load_model("base")
input_dir = "videos"
output_file = "video_transcripts.json"

transcripts = {}

for filename in os.listdir(input_dir):
    if filename.endswith(".webm"):
        path = os.path.join(input_dir, filename)
        print(f"Transcribing: {filename}")
        result = model.transcribe(path)
        transcripts[filename] = result['text']

# Save to JSON
with open(output_file, "w") as f:
    json.dump(transcripts, f, indent=2)

print(f"Saved all transcripts to {output_file}")
