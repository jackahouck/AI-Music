import glob, os, numpy as np, pretty_midi as pm, argparse, joblib

SEQ_LEN = 16 

def first_pitch_sequence(midipath):
    m = pm.PrettyMIDI(midipath)
    notes = []
    for inst in m.instruments:
        if inst.is_drum: 
            continue
        for n in inst.notes:
            notes.append((n.start, n.pitch))
    notes.sort(key=lambda x: x[0])
    return np.array([p for _, p in notes], dtype=np.int16)

def seed_from_mood(mood, raw_root="data/mood/raw"):
    files = glob.glob(os.path.join(raw_root, mood, "*.mid"))
    if not files:
        raise RuntimeError(f"No MIDI files for mood '{mood}' in {raw_root}/{mood}")
    seq = first_pitch_sequence(np.random.choice(files))
    if len(seq) < SEQ_LEN:
        raise RuntimeError("Chosen file too short for seed window.")
    start = np.random.randint(0, len(seq) - SEQ_LEN + 1)
    return (seq[start:start+SEQ_LEN] % 128).astype(np.int16)

def generate_sequence(model, seed, length=50):
    notes = list(seed)

    for i in range(length):
        probs = model.predict_proba(seed.reshape(1, -1))[0]
        next_note = np.random.choice(model.classes_, p=probs)
        notes.append(next_note)
        seed = np.append(seed[1:], next_note)

    return notes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mood", type=str, required=True, choices=["calm","energetic","somber","upbeat"])
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dur", type=float, default=0.45)
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    model = joblib.load("models/baseline.pkl")
    seed = seed_from_mood(args.mood)

    notes = generate_sequence(model, seed, length=args.length)

    midi = pm.PrettyMIDI()
    piano = pm.Instrument(program=0)
    t = 0.0
    for p in notes:
        piano.notes.append(pm.Note(velocity=100, pitch=int(p), start=t, end=t+args.dur))
        t += args.dur
    midi.instruments.append(piano)

    os.makedirs("data/generated", exist_ok=True)
    out = args.outfile or f"data/generated/{args.mood}_sample.mid"
    midi.write(out)
    print("Wrote:", out)
