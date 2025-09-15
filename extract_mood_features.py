import os
import glob
import numpy as np
import pandas as pd
import pretty_midi as pm

MOODS = ['calm', 'energetic', 'somber', 'upbeat']
RAW_ROOT = 'data/mood/raw'
OUT_CSV = 'data/mood/features.csv'

def get_all_notes(midi: pm.PrettyMIDI):
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    return notes

def duration_seconds(midi: pm.PrettyMIDI):
    return float(midi.get_end_time())

def tempo_bpm(midi: pm.PrettyMIDI):
    times, tempo = midi.get_tempo_changes()
    if len(tempo):
        return float(np.median(tempo))
    
    #OR

    onsets = []
    for n in get_all_notes(midi):
        onsets.append(n.start)
    if len(onsets) < 2:
        return 0.0
    
    onsets = np.sort(np.array(onsets))
    iois = np.dff(onsets)
    med_ioi = float(np.median(iois))

    if med_ioi <= 0:
        return 0.0
    return 60 / med_ioi

def note_density(midi: pm.PrettyMIDI):
    dur = duration_seconds(midi)
    n = len(get_all_notes(midi))
    return float(n / dur) if dur > 0 else 0.0

def avg_velocity(midi: pm.PrettyMIDI):
    velos = [n.velocity for n in get_all_notes(midi)]
    return float(np.mean(velos)) if velos else 0.0

def pitch_range(midi: pm.PrettyMIDI):
    pitches = [n.pitch for n in get_all_notes(midi)]
    if not pitches:
        return 0
    return int(max(pitches) - min(pitches))

def ioi_variance(midi):
    onsets = [n.start for n in get_all_notes(midi)]
    if len(onsets) < 3:
        return 0.0
    onsets = np.sort(np.asarray(onsets))
    iois = np.diff(onsets)
    return float(np.var(iois)) if iois.size else 0.0

MAJOR = np.array([0,2,4,5,7,9,11])
MINOR = np.array([0,2,3,5,7,8,10])

def mode_major_prob(midi: pm.PrettyMIDI, window_s=30.0) -> float:
    # collect notes only from the opening window and ignore very short notes
    notes = [n for n in get_all_notes(midi)
             if n.start <= window_s and (n.end - n.start) >= 0.08]
    if not notes:
        return 0.5

    # duration and velocity-weighted pitch-class histogram
    pc = np.zeros(12, dtype=float)
    for n in notes:
        w = (max(1e-6, n.end - n.start) ** 1.5) * (n.velocity / 127.0)
        pc[n.pitch % 12] += w
    pc_sum = pc.sum()
    if pc_sum == 0:
        return 0.5
    pc /= pc_sum

    maj = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
    minr = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)
    maj /= maj.sum(); minr /= minr.sum()

    def best_sim(profile):
        best = 0.0
        for t in range(12):
            pr = np.roll(profile, t)
            num = float(np.dot(pc, pr))
            den = float(np.linalg.norm(pc) * np.linalg.norm(pr))
            best = max(best, (num / den) if den else 0.0)
        return best

    maj_score = best_sim(maj)
    min_score = best_sim(minr)
    denom = maj_score + min_score
    return float(maj_score / denom) if denom > 0 else 0.5

    
def features_for_file(path: str, modd: str):
    try:
        midi = pm.PrettyMIDI(path)
    except Exception as e:
        return {}
    return {
        "file": path,
        "mood": mood,
        "duration_s": duration_seconds(midi),
        "tempo_bpm": tempo_bpm(midi),
        "note_density": note_density(midi),
        "avg_velocity": avg_velocity(midi),
        "pitch_range": pitch_range(midi),
        "ioi_var": ioi_variance(midi),
        "mode_major_prob": mode_major_prob(midi),
    }

rows = []
for mood in MOODS:
    pattern = os.path.join(RAW_ROOT, mood, "*.mid")
    for path in glob.glob(pattern):
        row = features_for_file(path, mood)
        if row:
            rows.append(row)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(df)