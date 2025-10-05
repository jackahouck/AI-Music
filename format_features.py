import numpy as np
import pandas as pd
import pretty_midi as pm
import glob, os, json

SEQ_LEN = 16
DUR_BINS = np.array([0.25, 0.5, 1.0, 1.5, 2.0])  

def load_pitch_dur_beats(midipath):
    m = pm.PrettyMIDI(midipath)
    times, tempi = m.get_tempo_changes()
    bpm = float(np.median(tempi)) if len(tempi) else 120.0

    notes = []
    for inst in m.instruments:
        if inst.is_drum: 
            continue
        for n in inst.notes:
            dur_s = max(1e-3, n.end - n.start)
            dur_beats = dur_s * bpm / 60.0
            notes.append((n.start, int(n.pitch), dur_beats))
    if not notes:
        return np.array([]), bpm
    notes.sort(key=lambda x: x[0])
    pitches = np.array([p for _, p, _ in notes], dtype=np.int16)
    dbeats  = np.array([d for _, _, d in notes], dtype=np.float32)
    return np.stack([pitches, dbeats], axis=1), bpm  

def to_duration_class(dbeats):
    idx = np.argmin(np.abs(DUR_BINS.reshape(-1,1) - dbeats.reshape(1,-1)), axis=0)
    return idx.astype(np.int16)

def build_windows(seq_pd, seq_len=SEQ_LEN):
    T = len(seq_pd)
    X, y_pitch, y_dcls = [], [], []
    if T <= seq_len: 
        return np.zeros((0, seq_len*2), dtype=np.int16), np.zeros((0,), dtype=np.int16), np.zeros((0,), dtype=np.int16)
    pitches = seq_pd[:,0].astype(np.int16)
    dbeats  = seq_pd[:,1].astype(np.float32)
    dcls    = to_duration_class(dbeats)
    for i in range(T - seq_len):
        win_p = pitches[i:i+seq_len]
        win_d = dcls[i:i+seq_len]
        X.append(np.stack([win_p, win_d], axis=1).reshape(-1))  
        y_pitch.append(pitches[i+seq_len])
        y_dcls.append(dcls[i+seq_len])
    return np.array(X, dtype=np.int16), np.array(y_pitch, dtype=np.int16), np.array(y_dcls, dtype=np.int16)

if __name__ == "__main__":
    files = glob.glob("data/mood/raw/*/*.mid") or glob.glob("data/raw_midis/*.mid")
    X_list, yP_list, yD_list = [], [], []
    used = 0
    for f in files:
        seq, bpm = load_pitch_dur_beats(f)
        if seq.size == 0: 
            continue
        X, yP, yD = build_windows(seq, SEQ_LEN)
        if X.size:
            X_list.append(X); yP_list.append(yP); yD_list.append(yD); used += 1
    if not X_list:
        raise SystemExit("No training windows found.")

    X   = np.vstack(X_list)
    y_p = np.concatenate(yP_list)
    y_d = np.concatenate(yD_list)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_pd.npy", X)      
    np.save("data/processed/y_pitch.npy", y_p) 
    np.save("data/processed/y_dclass.npy", y_d)
    os.makedirs("models", exist_ok=True)
    np.save("models/duration_bins.npy", DUR_BINS)

    print("X:", X.shape, "y_pitch:", y_p.shape, "y_dclass:", y_d.shape, "files_used:", used)
