import pretty_midi as pm
import os

midi = pm.PrettyMIDI()
piano = pm.Instrument(program=0)

start = 0
duration = 0.5
pitches = [60,62,65,65,67,71,72]

for pitch in pitches:
    note = pm.Note(velocity=100, pitch=pitch, start=start, end=start + duration)
    piano.notes.append(note)
    start += duration

midi.instruments.append(piano)

os.makedirs("data/raw_midis", exist_ok=True)
midi.write("data/raw_midis/example.mid")