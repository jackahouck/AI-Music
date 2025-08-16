import numpy as np
import joblib
import pretty_midi as pm
from format_features import SEQ_LEN
import os

def generate_sequence(model, seed, length=50):
    notes = list(seed)

    for i in range(length):
        probs = model.predict_proba(seed.reshape(1, -1))[0]
        next_note = np.random.choice(model.classes_, p=probs)
        notes.append(next_note)
        seed = np.append(seed[1:], next_note)

    return notes

if __name__ == '__main__':
    model = joblib.load('models/baseline.pkl')

    X = np.load('data/processed/X.npy')
    seed = X[0].copy()

    generated_notes = generate_sequence(model, seed, length=100)

    midi = pm.PrettyMIDI()
    piano = pm.Instrument(program=0)

    start = 0
    duration = 0.5
    for pitch in generated_notes:
        note = pm.Note(velocity=100, pitch=int(pitch),
                       start=start, end=start + duration)
        piano.notes.append(note)
        start += duration

    midi.instruments.append(piano)
    os.makedirs('data/generated', exist_ok=True)
    midi.write('data/generated/generated.mid')

    print('Generated sequence saved to data/generated/generated.mid')