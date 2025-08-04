import pretty_midi as pm
import pandas as pd
import os

midi = pm.PrettyMIDI("data/raw_midis/example.mid")
notes = []

for instr in midi.instruments:
    if not instr.is_drum:
        for note in instr.notes:
            notes.append((note.start, note.pitch, note.end - note.start))

print(notes[:10])

os.makedirs("data/processed", exist_ok=True)
df = pd.DataFrame(notes, columns=["start", "pitch", "duration"])
df.to_csv("data/processed/notes.csv", index=False)