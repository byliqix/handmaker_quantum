"""
String Particles — tangan mengendalikan partikel dawai (string theory)
MediaPipe Hands + particle physics + vibrating strings + audio
"""
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time
import math
import random
import threading
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
TIP_IDS = [4, 8, 12, 16, 20]

# ─── Audio ───────────────────────────────────────────────────────────
SR = 44100
audio_lock = threading.Lock()
active_strings = {}
string_counter = 0

def audio_callback(outdata, frames, time_info, status):
    t = np.arange(frames) / SR
    out = np.zeros(frames)
    now = time.time()
    with audio_lock:
        dead = []
        for sid, s in active_strings.items():
            elapsed = now - s['start']
            if elapsed > s['dur']:
                dead.append(sid)
                continue
            lt = t + elapsed
            mask = lt < s['dur']
            fade = np.maximum(0, 1 - lt / s['dur'])
            wave = np.sin(2 * np.pi * s['freq'] * lt + s['phase'])
            wave += np.sin(2 * np.pi * s['freq'] * 2 * lt) * 0.15
            wave *= s['vol'] * fade
            wave[~mask] = 0
            out[:min(len(wave), frames)] += wave[:min(len(wave), frames)]
        for sid in dead:
            del active_strings[sid]
    out = np.tanh(out * 0.4) * 0.3
    outdata[:] = np.column_stack([out, out]).astype(np.float32)

output_stream = sd.OutputStream(
    samplerate=SR, channels=2, callback=audio_callback,
    blocksize=512, dtype=np.float32
)
output_stream.start()

def play_string(freq, vol=0.08, dur=0.4, phase=0):
    global string_counter
    with audio_lock:
        string_counter += 1
        active_strings[string_counter] = {
            'freq': freq, 'vol': vol,
            'start': time.time(), 'dur': dur, 'phase': phase
        }

def play_chord(notes, vol=0.05):
    for i, f in enumerate(notes):
        play_string(f, vol, 0.35 + i * 0.04)

# ─── Scales ─────────────────────────────────────────────────────────
SCALES = {
    'major':     [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
    'minor':     [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25],
    'pentatonic':[261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25],
    'chromatic': [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00],
    'bass':      [65.41, 73.42, 82.41, 98.00, 110.00, 130.81, 146.83, 196.00],
    'blues':     [261.63, 311.13, 329.63, 349.23, 392.00, 466.16, 523.25, 587.33],
}
current_scale = 'major'
SCALE_KEYS = list(SCALES.keys())
NOTE_NAMES = ['C','D','E','F','G','A','B','C']
FINGER_NAMES = ['thumb','index','middle','ring','pinky']
FINGER_HUES = [0, 45, 130, 210, 290]
# for fist detection: (tip_id, pip_id) per finger (excluding thumb)
FIST_CHECKS = [(8, 6), (12, 10), (16, 14), (20, 18)]

def hsv(h, s=200, v=200):
    c = cv2.cvtColor(np.uint8([[[int(h)%180, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(c[0]), int(c[1]), int(c[2]))

# ─── String Particle ────────────────────────────────────────────────
class StringParticle:
    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        self.idx = idx
        self.freq = SCALES[current_scale][idx % 8]
        self.hue = FINGER_HUES[idx % 5] + random.uniform(-10, 10)
        self.r = 5 + (idx % 4) * 1.5
        self.pulse = random.random() * math.pi * 2
        self.trail = deque(maxlen=10)
        self.id = random.randint(0, 999999)
        self.life = 1.0
        self.target_x = x
        self.target_y = y
        self.vibe = 0  # vibration intensity

    def update(self):
        self.vx *= 0.9
        self.vy *= 0.9
        self.pulse += 0.04 + self.vibe * 0.3

        # attract to target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        d = math.hypot(dx, dy)
        if d > 2:
            pull = 0.03 + self.vibe * 0.02
            self.vx += dx * pull
            self.vy += dy * pull

        self.x += self.vx
        self.y += self.vy
        self.x = max(5, min(635, self.x))
        self.y = max(5, min(475, self.y))
        self.trail.append((self.x, self.y))

# ─── App ────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        root.title("string particles · hand controlled")
        root.configure(bg='#0a0a14')

        self.label = tk.Label(root, borderwidth=0, highlightthickness=0)
        self.label.pack()
        self.info = tk.Label(root, text="", fg='#555', bg='#0a0a14',
                             font=('system-ui', 9))
        self.info.pack(pady=(2, 6))

        root.bind('<Key>', self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='/home/kingzhat/rdh/hand_landmarker.task'),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.particles = []
        self.max_particles = 10

        self.frame_no = 0
        self.cooldown = 0
        self.running = True
        self.prev_tips = {}
        self.was_fist = False
        self.explode_timer = 0
        self.explode_rings = []
        self.hand_landmarks_raw = []
        self.update()

    def update(self):
        if not self.running: return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update)
            return

        frame = cv2.flip(frame, 1)
        self.frame_no += 1
        t0 = time.time()
        h, w = 480, 640
        notes = SCALES[current_scale]

        # ─── MediaPipe ────────────────────────────────────────────
        all_hand_pts = []  # (x, y, landmark_idx, hand_idx)
        self.hand_landmarks_raw = []
        if self.frame_no % 2 == 0:
            small = cv2.resize(frame, (320, 240))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            result = self.landmarker.detect(mp_image)
            if result.hand_landmarks:
                for hand_idx, lm in enumerate(result.hand_landmarks):
                    self.hand_landmarks_raw.append(lm)
                    for fi in range(5):
                        li = TIP_IDS[fi]
                        all_hand_pts.append((
                            int(lm[li].x * w),
                            int(lm[li].y * h),
                            fi, hand_idx
                        ))

        # ─── Fist detection ───────────────────────────────────────
        is_fist = False
        for lm in self.hand_landmarks_raw:
            curled = 0
            # index, middle, ring, pinky: tip close to pip
            for tip_id, pip_id in FIST_CHECKS:
                d = math.hypot(lm[tip_id].x - lm[pip_id].x, lm[tip_id].y - lm[pip_id].y)
                if d < 0.06:
                    curled += 1
            # thumb tip close to index mcp (lm[5])
            thumb_d = math.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y)
            if thumb_d < 0.06:
                curled += 1
            if curled >= 4:
                is_fist = True
                break

        # trigger explosion on transition to fist
        if is_fist and not self.was_fist and self.particles:
            self.explode_timer = 45
            cx = sum(p.x for p in self.particles) / len(self.particles)
            cy = sum(p.y for p in self.particles) / len(self.particles)
            for p in self.particles:
                angle = math.atan2(p.y - cy, p.x - cx) + random.uniform(-0.3, 0.3)
                speed = random.uniform(5, 15)
                p.vx += math.cos(angle) * speed
                p.vy += math.sin(angle) * speed
                p.vibe = 1.0
                p.trail.clear()
            for _ in range(5):
                r = random.uniform(20, 120)
                self.explode_rings.append({
                    'cx': cx + random.uniform(-30, 30),
                    'cy': cy + random.uniform(-30, 30),
                    'r': r, 'max_r': r + 120, 'alpha': 1.0
                })
            play_string(110, 0.2, 0.6, phase=0)
            play_string(55, 0.1, 0.8, phase=0.3)
        self.was_fist = is_fist

        # ─── Camera background ────────────────────────────────────
        display = cv2.convertScaleAbs(frame, alpha=1.0, beta=10)

        # ─── Particle targets from hands ──────────────────────────
        if all_hand_pts:
            # spawn one particle per fingertip
            while len(self.particles) < len(all_hand_pts):
                idx = len(self.particles)
                hp = all_hand_pts[idx]
                p = StringParticle(hp[0], hp[1], idx)
                p.r = 8
                self.particles.append(p)

            # kill excess particles
            self.particles = self.particles[:len(all_hand_pts)]

            # update particles
            for pi, p in enumerate(self.particles):
                hp = all_hand_pts[pi]
                tx, ty, fi, hand_idx = hp

                p.target_x = tx
                p.target_y = ty
                p.hue = FINGER_HUES[fi % 5] + hand_idx * 30
                p.freq = notes[fi % 8]
                p.life = min(1.0, p.life + 0.05)

                key = f"{hand_idx}_{fi}"
                prev = self.prev_tips.get(key)
                if prev:
                    moved = math.hypot(tx - prev[0], ty - prev[1])
                    p.vibe = min(1.0, moved * 0.03)
                    if moved > 10:
                        vol = min(0.10, 0.02 + moved * 0.0015)
                        play_string(p.freq, vol, 0.2 + moved * 0.003)
                self.prev_tips[key] = (tx, ty)

            if len(all_hand_pts) >= 5 and random.random() < 0.008:
                play_chord([notes[random.randint(0, 7)] for _ in range(3)], 0.03)
        else:
            # fade & remove dead particles
            self.particles = [p for p in self.particles if p.life > 0.05]
            for p in self.particles:
                p.life -= 0.03
                p.target_x = 320 + math.sin(p.idx * 2.1 + self.frame_no * 0.01) * 150
                p.target_y = 240 + math.cos(p.idx * 1.7 + self.frame_no * 0.01) * 120
                p.vibe *= 0.95
            self.prev_tips.clear()

        # ─── explosion timer ──────────────────────────────────────
        if self.explode_timer > 0:
            self.explode_timer -= 1

        # ─── update ───────────────────────────────────────────────
        for p in self.particles:
            p.update()

        # ─── string connections ───────────────────────────────────
        conns = []
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                a, b = self.particles[i], self.particles[j]
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < 200:
                    conns.append((a, b, 1 - d / 200))

        self.cooldown -= 1
        if self.cooldown <= 0 and conns:
            for a, b, intens in sorted(conns, key=lambda x: -x[2])[:4]:
                vibe = (a.vibe + b.vibe) / 2 * intens
                if vibe > 0.05 and random.random() < vibe * 0.3:
                    freq = (a.freq + b.freq) / 2
                    play_string(freq, vibe * 0.05, 0.25)
            self.cooldown = 3

        # ─── draw strings ─────────────────────────────────────────
        seen = set()
        for a, b, intens in conns:
            pid = tuple(sorted([a.id, b.id]))
            if pid in seen: continue
            seen.add(pid)

            vibe = (a.vibe + b.vibe) / 2 * intens
            hue = (a.hue + b.hue) / 2
            color = hsv(hue % 180, 180, 200)
            glow = tuple(int(v * 0.15) for v in color)
            pt1, pt2 = (int(a.x), int(a.y)), (int(b.x), int(b.y))
            d = math.hypot(a.x - b.x, a.y - b.y)

            if vibe > 0.05:
                steps = max(8, int(d / 8))
                pts = []
                for k in range(steps + 1):
                    t = k / steps
                    mx = a.x + (b.x - a.x) * t
                    my = a.y + (b.y - a.y) * t
                    px = -(b.y - a.y) / max(d, 1)
                    py = (b.x - a.x) / max(d, 1)
                    wave = math.sin(t * math.pi * 3 + self.frame_no * 0.08) * vibe * 10
                    mx += px * wave
                    my += py * wave
                    pts.append((int(mx), int(my)))

                for k in range(1, len(pts)):
                    cv2.line(display, pts[k-1], pts[k], glow, 2, cv2.LINE_AA)
                for k in range(1, len(pts)):
                    cv2.line(display, pts[k-1], pts[k], color, 1, cv2.LINE_AA)
            else:
                cv2.line(display, pt1, pt2, glow, 2, cv2.LINE_AA)
                cv2.line(display, pt1, pt2, color, 1, cv2.LINE_AA)

        # ─── trails ───────────────────────────────────────────────
        for p in self.particles:
            base = hsv(p.hue % 180, 180, 180)
            for i in range(1, len(p.trail)):
                a = (i / len(p.trail)) * 0.3
                c = tuple(int(v * a) for v in base)
                cv2.line(display, (int(p.trail[i-1][0]), int(p.trail[i-1][1])),
                         (int(p.trail[i][0]), int(p.trail[i][1])), c, 1, cv2.LINE_AA)

        # ─── particles ─────────────────────────────────────────────
        for p in self.particles:
            pr = math.sin(p.pulse) * 0.2 + 1
            r = max(2, int(p.r * pr))
            color = hsv(p.hue % 180, 200, 220)
            pt = (int(p.x), int(p.y))

            for gr in [r * 3, r * 2]:
                gc = tuple(int(v * 0.06) for v in color)
                cv2.circle(display, pt, gr, gc, -1, cv2.LINE_AA)
            cv2.circle(display, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(display, pt, r, color, -1, cv2.LINE_AA)

            if p.life > 0.5:
                cv2.putText(display, NOTE_NAMES[p.idx % 8],
                           (pt[0] - 7, pt[1] - r - 8),
                           cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)

        # ─── explosion rings ──────────────────────────────────────
        if self.explode_timer > 0:
            for ring in self.explode_rings:
                ring['r'] += 4
                ring['alpha'] = max(0, 1 - (ring['r'] - 20) / ring['max_r'])
                if ring['alpha'] > 0.05:
                    c = hsv(int(self.frame_no * 2) % 180, 200, int(255 * ring['alpha']))
                    cv2.circle(display, (int(ring['cx']), int(ring['cy'])),
                               int(ring['r']), c, 2, cv2.LINE_AA)
            self.explode_rings = [r for r in self.explode_rings if r['alpha'] > 0.05]

        # ─── hand landmark glow ───────────────────────────────────
        for tx, ty, li, hand_idx in all_hand_pts:
            c = hsv((li * 12 + hand_idx * 40) % 180, 200, 200)
            cv2.circle(display, (tx, ty), 4, c, 1, cv2.LINE_AA)

        # ─── HUD ──────────────────────────────────────────────────
        fps = 1 / (time.time() - t0 + 0.001)
        num_hands = len(set(hi for _,_,_,hi in all_hand_pts)) if all_hand_pts else 0
        cv2.putText(display, f"tangan: {num_hands}  scale: {current_scale}",
                    (14, 26), cv2.FONT_HERSHEY_DUPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(display, f"strings: {len(seen)}  fps: {fps:.0f}",
                    (14, 48), cv2.FONT_HERSHEY_DUPLEX, 0.32, (130, 130, 130), 1)
        cv2.putText(display, "1-6: scale  |  q: quit",
                    (14, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.28, (60, 60, 60), 1)

        # ─── tkinter ──────────────────────────────────────────────
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.info.config(text=f"{num_hands} tangan · {current_scale} · {len(seen)} strings · {fps:.0f} fps")
        self.root.after(30, self.update)

    def on_key(self, event):
        global current_scale
        k = event.char
        if k in ('q', 'Q'):
            self.running = False
            self.cap.release()
            self.landmarker.close()
            output_stream.stop()
            self.root.destroy()
            return
        if k in '123456':
            current_scale = SCALE_KEYS[int(k) - 1]
            for p in self.particles:
                p.freq = SCALES[current_scale][p.idx % 8]
            play_string(440, 0.06, 0.3)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.landmarker.close()
        output_stream.stop()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
