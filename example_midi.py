import pretty_midi as pm
import os
import random

#Random walk params
NUM_NOTES = 200
START_PITCH = 60
DURATION = 0.5
STEP_RANGE = [-2, -1, 0, 1, 2]

midi = pm.PrettyMIDI()
piano = pm.Instrument(program=0)

start = 0
current_pitch = START_PITCH

for _ in range(NUM_NOTES):
    note = pm.Note(
        velocity=100, pitch=current_pitch, start=start, end=start + DURATION
    )
    piano.notes.append(note)
    start += DURATION

    step = random.choice(STEP_RANGE)
    current_pitch = max(21, min(108, current_pitch + step))

midi.instruments.append(piano)
os.makedirs("data/raw_midis", exist_ok=True)
midi.write("data/raw_midis/example.mid")