import numpy as np

def generate_sequence(model, seed, length=50):
    notes = list(seed)

    for i in range(length):
        probs = model.predict_proba(seed.reshape(1, -1))[0]
        next_note = np.random.choice(model.classes_, p=probs)
        notes.append(next_note)
        seed = np.append(seed[1:], next_note)

    return notes