import glob, os, numpy as np, pretty_midi as pm, argparse, joblib

SEQ_LEN = 16 

def first_pitch_dclass_sequence(midipath):
    m = pm.PrettyMIDI(midipath)
    times, tempi = m.get_tempo_changes()
    bpm = float(np.median(tempi)) if len(tempi) else 120.0

    notes = []
    for inst in m.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            dur_beats = max(1e-3, n.end - n.start) * bpm / 60.0
            notes.append((n.start, int(n.pitch), dur_beats))
    if not notes:
        raise RuntimeError("No notes in seed file.")
    notes.sort(key=lambda x: x[0])
    pitches = np.array([p for _, p, _ in notes], dtype=np.int16)
    dbeats  = np.array([d for _, _, d in notes], dtype=np.float32)
    idx = np.argmin(np.abs(DUR_BINS.reshape(-1,1) - dbeats.reshape(1,-1)), axis=0).astype(np.int16)
    return pitches, idx, bpm


def seed_from_mood_pd(mood):
    files = glob.glob(os.path.join("data/mood/raw", mood, "*.mid"))
    if not files:
        raise RuntimeError(f"No files for mood={mood}")
    f = np.random.choice(files)
    P, Dcls, bpm = first_pitch_dclass_sequence(f)
    if len(P) < SEQ_LEN:
        raise RuntimeError("Seed file too short.")
    start = np.random.randint(0, len(P) - SEQ_LEN + 1)
    win_p = P[start:start+SEQ_LEN]
    win_d = Dcls[start:start+SEQ_LEN]
    seed_vec = np.stack([win_p, win_d], axis=1).reshape(-1)  
    return seed_vec, bpm

def generate_sequence_pd(model, seed_vec, length, bpm):
    pitch_classes = np.arange(128, dtype=np.int16)
    dur_classes   = np.arange(len(DUR_BINS), dtype=np.int16)

    vec = seed_vec.copy()
    out_p, out_d = [], []
    for _ in range(length):
        p, dcls = predict_next(model, vec)
        out_p.append(p); out_d.append(dcls)
        vec = np.concatenate([vec[2:], np.array([p, dcls], dtype=np.int16)])
    d_beats = DUR_BINS[np.array(out_d)]
    d_secs  = d_beats * (60.0 / bpm)
    return np.array(out_p), d_secs


def predict_next(model, seed_vec):
    proba_list = model.predict_proba(seed_vec.reshape(1, -1))  
    pp = proba_list[0][0]  
    pd = proba_list[1][0] 

    pitch_classes = model.estimators_[0].classes_
    dur_classes   = model.estimators_[1].classes_

    if not np.isfinite(pp).all() or pp.sum() <= 0:
        pp = np.ones_like(pp) / len(pp)
    if not np.isfinite(pd).all() or pd.sum() <= 0:
        pd = np.ones_like(pd) / len(pd)

    next_pitch = int(np.random.choice(pitch_classes, p=pp))
    next_dcls  = int(np.random.choice(dur_classes,   p=pd))
    return next_pitch, next_dcls


DUR_BINS = np.load("models/duration_bins.npy")

def write_midi(pitches, dsecs, outfile, velocity=100):
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    t = 0.0
    for p, d in zip(pitches, dsecs):
        p = int(max(21, min(108, p)))
        inst.notes.append(pm.Note(velocity=velocity, pitch=p, start=t, end=t+float(max(0.06, d))))
        t += float(max(0.06, d))
    midi.instruments.append(inst)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    midi.write(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mood", type=str, required=True, choices=["calm","energetic","somber","upbeat"])
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    model = joblib.load("models/baseline_pd.pkl")
    DUR_BINS = np.load("models/duration_bins.npy")

    seed_vec, bpm = seed_from_mood_pd(args.mood)

    pitches, dsecs = generate_sequence_pd(model, seed_vec, length=args.length, bpm=bpm)

    os.makedirs("data/generated", exist_ok=True)
    out = args.outfile or f"data/generated/{args.mood}_pd.mid"
    write_midi(pitches, dsecs, out)
    print("Wrote:", out)