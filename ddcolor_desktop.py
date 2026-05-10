"""
DDColor Lab — Aplicación de escritorio
=======================================
Coloca los pesos en ./models/:
    models/ddcolor_comic.pth        (tu fine-tune, guardado por BasicSR)
    models/ddcolor_realistic.pth    (modelo oficial ModelScope / HuggingFace)
    models/ddcolor_artistic.pth     (modelo artístico ModelScope / HuggingFace)

    Ambos pueden ser 'large' (convnext-l) o 'tiny' (convnext-t).
    La app detecta el tamaño automáticamente por el nombre del fichero.

Requiere:
    pip install pillow scikit-image opencv-python torch torchvision lpips
    + el paquete DDColor instalado (python setup.py develop)
"""

import os
import cv2
import threading
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk, ImageDraw

warnings.filterwarnings("ignore")

# ── Dependencias opcionales ───────────────────────────────────────────────────
try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import lpips as lpips_lib
    _lpips_fn = None  # se inicializa lazy
    LPIPS_OK = True
except ImportError:
    LPIPS_OK = False

# ── Paleta minimalista ────────────────────────────────────────────────────────
BG       = "#111318"
BG2      = "#16191f"
BG3      = "#1c2028"
BORDER   = "#262b36"
TEXT     = "#b8bcc8"
TEXT_HI  = "#eef0f5"
TEXT_DIM = "#50586a"
ACCENT   = "#4f7aff"
GREEN    = "#3dba8a"
AMBER    = "#e8a535"
RED      = "#e05a5a"
COMIC    = "#4f7aff"
REAL     = "#9d7fe8"
ART      = "#ff7a4f"

MODELS_DIR = Path("./models")
PREVIEW_W  = 360
PREVIEW_H  = 280
THUMB_W    = 160
THUMB_H    = 120

MODEL_STYLES = ("comic", "realistic", "artistic")
MODEL_FILES  = {
    "comic":     "ddcolor_comic.pth",
    "realistic": "ddcolor_realistic.pth",
    "artistic":  "ddcolor_artistic.pth",
}


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline de inferencia
# ═════════════════════════════════════════════════════════════════════════════

def detect_model_size(path: Path) -> str:
    return "tiny" if "tiny" in path.stem.lower() else "large"


def try_load_pipeline(path: Path, input_size: int = 512):
    if not path.exists():
        return None
    try:
        from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_ddcolor_model(
            DDColor,
            model_path=str(path),
            input_size=input_size,
            model_size=detect_model_size(path),
            device=device,
        )
        return ColorizationPipeline(model, input_size=input_size, device=device)
    except Exception as e:
        print(f"[ERROR] {path.name}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Métricas: Delta CF · Delta E · LPIPS
# ═════════════════════════════════════════════════════════════════════════════

def _colorfulness(img_bgr: np.ndarray) -> float:
    """Hasler & Süsstrunk colorfulness score."""
    img = img_bgr.astype(float)
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    rg = R - G
    yb = 0.5 * (R + G) - B
    return float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )


def metric_delta_cf(pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> float | None:
    """ΔCF = |CF(pred) − CF(gt)|.  Más cercano a 0 → viveza cromática similar al GT."""
    if gt_bgr is None:
        return None
    return abs(_colorfulness(pred_bgr) - _colorfulness(gt_bgr))


def metric_delta_e(pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> float | None:
    """ΔE 2000 medio píxel a píxel, implementación vectorizada en NumPy. ↓ mejor."""
    if gt_bgr is None:
        return None

    gt_rs    = cv2.resize(gt_bgr, (pred_bgr.shape[1], pred_bgr.shape[0]))
    pred_lab = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    gt_lab   = cv2.cvtColor(gt_rs,   cv2.COLOR_BGR2Lab).astype(np.float32)

    # Rescalar de rango OpenCV a CIE: L∈[0,100], a/b∈[-128, 127]
    for lab in (pred_lab, gt_lab):
        lab[:, :, 0]  = lab[:, :, 0] * (100.0 / 255.0)
        lab[:, :, 1:] = lab[:, :, 1:] - 128.0

    # ΔE 2000 vectorizado en numpy (implementación completa, sin dependencias extra)
    L1, a1, b1 = pred_lab[:,:,0], pred_lab[:,:,1], pred_lab[:,:,2]
    L2, a2, b2 = gt_lab[:,:,0],   gt_lab[:,:,1],   gt_lab[:,:,2]

    # Paso 1: croma
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0

    # Paso 2: factor G y ajuste de a*
    C_avg7 = C_avg ** 7
    G = 0.5 * (1.0 - np.sqrt(C_avg7 / (C_avg7 + 25.0**7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)    # Paso 3: nuevo croma y tono
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    # Paso 4: diferencias
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(np.abs(dhp) > 180, dhp - 360 * np.sign(dhp), dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    # Paso 5: medias
    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0
    hp_avg = h1p + h2p
    hp_avg = np.where(np.abs(h1p - h2p) > 180,
                      np.where(hp_avg < 360, hp_avg + 360, hp_avg - 360),
                      hp_avg) / 2.0

    # Paso 5: factores S y T
    T = (1.0
         - 0.17 * np.cos(np.radians(hp_avg - 30))
         + 0.24 * np.cos(np.radians(2 * hp_avg))
         + 0.32 * np.cos(np.radians(3 * hp_avg + 6))
         - 0.20 * np.cos(np.radians(4 * hp_avg - 63)))
    SL = 1.0 + 0.015 * (Lp_avg - 50)**2 / np.sqrt(20 + (Lp_avg - 50)**2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    # Paso 6: término de rotación RT
    d_theta = 30 * np.exp(-((hp_avg - 275) / 25)**2)
    Cp_avg7 = Cp_avg ** 7
    RC = 2.0 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25.0**7))
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    # Paso 7: ΔE2000
    de = np.sqrt(
        (dLp / SL)**2 +
        (dCp / SC)**2 +
        (dHp / SH)**2 +
        RT * (dCp / SC) * (dHp / SH)
    )
    return float(np.mean(de))


def _get_lpips_fn():
    global _lpips_fn
    if not LPIPS_OK:
        return None
    if _lpips_fn is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
        _lpips_fn.eval()
    return _lpips_fn


def metric_lpips(pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> float | None:
    """LPIPS (AlexNet). ↓ mejor. Requiere torch + lpips."""
    if gt_bgr is None or not TORCH_OK or not LPIPS_OK:
        return None
    try:
        fn = _get_lpips_fn()
        device = next(fn.parameters()).device

        def to_tensor(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        gt_rs = cv2.resize(gt_bgr, (pred_bgr.shape[1], pred_bgr.shape[0]))
        with torch.no_grad():
            score = fn(to_tensor(pred_bgr), to_tensor(gt_rs))
        return float(score.item())
    except Exception as e:
        print(f"[LPIPS error] {e}")
        return None


def compute_metrics(pred_bgr: np.ndarray, gt_bgr) -> dict:
    return {
        "delta_cf":  metric_delta_cf(pred_bgr, gt_bgr),
        "delta_e":   metric_delta_e(pred_bgr,  gt_bgr),
        "lpips":     metric_lpips(pred_bgr,    gt_bgr),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Helpers UI
# ═════════════════════════════════════════════════════════════════════════════

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def fit_image(img_pil: Image.Image, w: int, h: int) -> ImageTk.PhotoImage:
    canvas = Image.new("RGB", (w, h), BG2)
    img_pil.thumbnail((w, h), Image.LANCZOS)
    ox = (w - img_pil.width) // 2
    oy = (h - img_pil.height) // 2
    canvas.paste(img_pil, (ox, oy))
    return ImageTk.PhotoImage(canvas)


def bgr_to_photoimage(img_bgr: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    return fit_image(bgr_to_pil(img_bgr), w, h)


def placeholder(w: int, h: int, text: str) -> ImageTk.PhotoImage:
    img = Image.new("RGB", (w, h), BG3)
    ImageDraw.Draw(img).text((w // 2, h // 2), text, fill=TEXT_DIM, anchor="mm")
    return ImageTk.PhotoImage(img)


def section_bar(parent, text: str):
    f = tk.Frame(parent, bg=BG3)
    f.pack(fill="x", pady=(10, 2))
    tk.Label(f, text=f"  {text}", bg=BG3, fg=TEXT_DIM,
             font=("Helvetica", 9, "bold"), anchor="w", pady=4).pack(fill="x")


def sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=0, pady=5)


def flat_button(parent, text, command, bg=BG3, fg=TEXT, bold=False, pady=5):
    font = ("Helvetica", 9, "bold") if bold else ("Helvetica", 9)
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, relief="flat", font=font, pady=pady,
        cursor="hand2", activebackground=BORDER, activeforeground=TEXT_HI,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Aplicación principal
# ═════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    # ── Definición de métricas ────────────────────────────────────────────────
    METRIC_DEFS = [
        # (etiqueta, clave, lower_is_better, descripción)
        ("ΔCF  ↓",              "delta_cf", True,
         "|CF(pred) − CF(GT)|. Cercano a 0 = viveza cromática similar al GT.  Requiere GT."),
        ("ΔE 2000  ↓",          "delta_e",  True,
         "Error colorimétrico perceptual ΔE2000 medio píxel a píxel en CIELAB.  Requiere GT."),
        ("LPIPS  ↓",            "lpips",    True,
         "Perceptual similarity (AlexNet). ↓ = más parecido al GT.  Requiere GT + lpips."),
    ]

    def __init__(self):
        super().__init__()
        self.title("DDColor Lab")
        self.configure(bg=BG)
        self.minsize(1020, 700)

        self._input_bgr = None
        self._gt_bgr    = None
        self._results   = {s: None for s in MODEL_STYLES}   # bgr outputs
        self._metrics   = {s: {}   for s in MODEL_STYLES}
        self._pipes     = {s: None for s in MODEL_STYLES}

        MODELS_DIR.mkdir(exist_ok=True)
        self._build_ui()
        self._load_models_bg()

    # ─────────────────────────────────────────────────────────────────────────
    # Construcción de UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_topbar()
        nb = self._build_notebook()
        tabs = [tk.Frame(nb, bg=BG) for _ in range(3)]
        for tab, label in zip(tabs, ["  Colorizar y comparar  ", "  Métricas  ", "  Guía  "]):
            nb.add(tab, text=label)
        self._build_main(tabs[0])
        self._build_metrics_tab(tabs[1])
        self._build_guide(tabs[2])

    def _build_topbar(self):
        top = tk.Frame(self, bg=BG2, pady=10)
        top.pack(fill="x")
        tk.Frame(top, bg=BORDER, height=1).pack(fill="x", side="bottom")
        tk.Label(top, text="DDColor Lab", bg=BG2, fg=TEXT_HI,
                 font=("Helvetica", 14, "bold"), padx=16).pack(side="left")
        tk.Label(top, text="EVALUACIÓN DE COLORIZACIÓN · ICCV 2023",
                 bg=BG2, fg=TEXT_DIM, font=("Helvetica", 8)).pack(side="left")
        self._var_status = tk.StringVar(value="Cargando modelos…")
        tk.Label(top, textvariable=self._var_status, bg=BG2,
                 fg=GREEN, font=("Helvetica", 9), padx=16).pack(side="right")

    def _build_notebook(self) -> ttk.Notebook:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("T.TNotebook",     background=BG,  borderwidth=0)
        style.configure("T.TNotebook.Tab", background=BG3, foreground=TEXT_DIM,
                        font=("Helvetica", 9), padding=[14, 5])
        style.map("T.TNotebook.Tab",
                  background=[("selected", BG2)],
                  foreground=[("selected", ACCENT)])
        nb = ttk.Notebook(self, style="T.TNotebook")
        nb.pack(fill="both", expand=True)
        return nb

    # ── Tab 1: Colorizar ──────────────────────────────────────────────────────

    def _build_main(self, parent):
        left = tk.Frame(parent, bg=BG2, width=210)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Frame(parent, bg=BORDER, width=1).pack(side="left", fill="y")
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_sidebar(left)
        self._build_results_panel(right)

    def _build_sidebar(self, parent):
        # Entrada
        section_bar(parent, "ENTRADA")
        frm = tk.Frame(parent, bg=BG3, width=THUMB_W, height=THUMB_H)
        frm.pack(pady=6)
        frm.pack_propagate(False)
        self._ph_input = placeholder(THUMB_W, THUMB_H, "Abrir imagen")
        self._lbl_input = tk.Label(frm, image=self._ph_input, bg=BG3, cursor="hand2")
        self._lbl_input.pack(expand=True)
        self._lbl_input.bind("<Button-1>", lambda _: self._open_input())
        flat_button(parent, "Abrir imagen", self._open_input,
                    bg=ACCENT, fg="white", bold=True, pady=6).pack(fill="x", padx=12, pady=(2, 0))

        # Ground Truth
        section_bar(parent, "GROUND TRUTH")
        flat_button(parent, "Abrir GT (opcional)", self._open_gt,
                    pady=5).pack(fill="x", padx=12, pady=2)
        self._lbl_gt = tk.Label(parent, text="Auto (misma imagen)",
                                bg=BG2, fg=GREEN, font=("Helvetica", 8))
        self._lbl_gt.pack(pady=2)

        # Modelos
        section_bar(parent, "MODELOS")
        self._model_vars = {
            "comic":     tk.BooleanVar(value=True),
            "realistic": tk.BooleanVar(value=True),
            "artistic":  tk.BooleanVar(value=True),
        }
        labels = {"comic": ("Cómic FT", COMIC), "realistic": ("Realista HF", REAL),
                  "artistic": ("Artístico HF", ART)}
        for style, (txt, col) in labels.items():
            tk.Checkbutton(
                parent, variable=self._model_vars[style], text=txt,
                bg=BG2, fg=col, selectcolor=BG3,
                activebackground=BG2, activeforeground=col,
                font=("Helvetica", 9), bd=0,
            ).pack(anchor="w", padx=16, pady=2)

        sep(parent)

        # Botones acción
        self._btn_run = flat_button(parent, "▶  Ejecutar", self._run,
                                    bg=ACCENT, fg="white", bold=True, pady=8)
        self._btn_run.pack(fill="x", padx=12, pady=4)
        self._btn_save = flat_button(parent, "Guardar resultados", self._save, pady=5)
        self._btn_save.configure(state="disabled")
        self._btn_save.pack(fill="x", padx=12, pady=2)

        self._progress = ttk.Progressbar(parent, mode="indeterminate")
        self._progress.pack(fill="x", padx=12, pady=4)

        self._lbl_model_info = tk.Label(
            parent, text="", bg=BG2, fg=TEXT_DIM,
            font=("Helvetica", 8), wraplength=185, justify="left",
        )
        self._lbl_model_info.pack(padx=12, pady=6, anchor="w")

    def _build_results_panel(self, parent):
        results = tk.Frame(parent, bg=BG)
        results.pack(fill="both", expand=True, padx=12, pady=10)
        results.columnconfigure(0, weight=1)
        results.columnconfigure(1, weight=1)
        results.rowconfigure(0, weight=1)

        # Columna izquierda: Cómic
        comic_col = tk.Frame(results, bg=BG)
        comic_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(comic_col, text="Modelo Cómic FT", bg=BG, fg=COMIC,
                 font=("Helvetica", 9, "bold"), anchor="w").pack(fill="x", pady=(0, 4))
        self._frm_comic = tk.Frame(comic_col, bg=BG3)
        self._frm_comic.pack(fill="both", expand=True)
        self._ph_comic = placeholder(PREVIEW_W, PREVIEW_H, "Modelo Cómic FT")
        self._lbl_comic = tk.Label(self._frm_comic, image=self._ph_comic, bg=BG3)
        self._lbl_comic.pack(fill="both", expand=True)

        # Columna derecha: Notebook Real / Art
        right_col = tk.Frame(results, bg=BG)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        side_nb = ttk.Notebook(right_col, style="T.TNotebook")
        side_nb.pack(fill="both", expand=True)

        self._lbl_real, self._ph_real = self._add_preview_tab(
            side_nb, "  Realista HF  ", "Modelo Realista HF", REAL)
        self._lbl_art, self._ph_art = self._add_preview_tab(
            side_nb, "  Artístico HF  ", "Modelo Artístico HF", ART)

        # Barra de resumen
        mbar = tk.Frame(parent, bg=BG2, pady=5)
        mbar.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(mbar, text="RESUMEN", bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 8, "bold"), padx=8).pack(side="left")
        self._lbl_summary = tk.Label(
            mbar, text="Ejecuta la colorización para ver los resultados.",
            bg=BG2, fg=TEXT_DIM, font=("Helvetica", 9), padx=8,
        )
        self._lbl_summary.pack(side="left")

    def _add_preview_tab(self, nb, tab_label, title, color):
        f = tk.Frame(nb, bg=BG)
        nb.add(f, text=tab_label)
        tk.Label(f, text=title, bg=BG, fg=color,
                 font=("Helvetica", 9, "bold"), anchor="w").pack(fill="x", pady=(0, 4))
        frm = tk.Frame(f, bg=BG3)
        frm.pack(fill="both", expand=True)
        ph = placeholder(PREVIEW_W, PREVIEW_H, title)
        lbl = tk.Label(frm, image=ph, bg=BG3)
        lbl.pack(fill="both", expand=True)
        return lbl, ph

    # ── Tab 2: Métricas ───────────────────────────────────────────────────────

    def _build_metrics_tab(self, parent):
        tk.Label(
            parent,
            text="Métricas de la última colorización.\n"
                 "Todas requieren Ground Truth (se usa la imagen de entrada por defecto).",
            bg=BG, fg=TEXT_DIM, font=("Helvetica", 9),
            justify="left", pady=8,
        ).pack(anchor="w", padx=20)

        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=4)

        headers = [("Métrica", 24), ("Cómic FT", 13), ("Realista HF", 13),
                   ("Artístico", 13), ("↑/↓", 5), ("Descripción", 38)]
        for c, (h, w) in enumerate(headers):
            tk.Label(frame, text=h, bg=BG3, fg=TEXT_DIM,
                     font=("Helvetica", 8, "bold"), width=w, anchor="w",
                     padx=6, pady=5, relief="flat",
                     ).grid(row=0, column=c, sticky="ew", padx=1, pady=(0, 2))

        self._metric_rows = []
        for r, (name, key, lower, desc) in enumerate(self.METRIC_DEFS):
            row = []
            for c, (txt, w, fg) in enumerate([
                (name, 24, TEXT),
                ("—",  13, TEXT_DIM),
                ("—",  13, TEXT_DIM),
                ("—",  13, TEXT_DIM),
                ("",    5, TEXT_DIM),
                (desc, 38, TEXT_DIM),
            ]):
                lbl = tk.Label(frame, text=txt, bg=BG2, fg=fg,
                               font=("Helvetica", 9), width=w, anchor="w",
                               padx=6, pady=5, relief="flat")
                lbl.grid(row=r + 1, column=c, sticky="ew", padx=1, pady=1)
                row.append(lbl)
            self._metric_rows.append(row)

        for c in range(6):
            frame.columnconfigure(c, weight=1 if c == 5 else 0)

        leg = tk.Frame(parent, bg=BG)
        leg.pack(anchor="w", padx=20, pady=10)
        for txt, col in [("✓ mejor resultado", GREEN),
                         ("(GT) todas las métricas requieren ground truth", TEXT_DIM)]:
            tk.Label(leg, text=txt, bg=BG, fg=col,
                     font=("Helvetica", 9)).pack(side="left", padx=12)

    # ── Tab 3: Guía ───────────────────────────────────────────────────────────

    def _build_guide(self, parent):
        txt = tk.Text(parent, bg=BG2, fg=TEXT, font=("Helvetica", 10),
                      relief="flat", padx=20, pady=16, wrap="word",
                      insertbackground=ACCENT, selectbackground=ACCENT)
        sb = tk.Scrollbar(parent, command=txt.yview, bg=BG3, troughcolor=BG)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)

        txt.tag_configure("h1",   font=("Helvetica", 13, "bold"), foreground=TEXT_HI)
        txt.tag_configure("h2",   font=("Helvetica", 10, "bold"), foreground=ACCENT)
        txt.tag_configure("code", font=("Courier", 9), foreground="#22d3ee", background=BG3)
        txt.tag_configure("dim",  foreground=TEXT_DIM)
        txt.tag_configure("ok",   foreground=GREEN)
        txt.tag_configure("amber", foreground=AMBER)

        def w(t, *tags):
            txt.insert("end", t, tags)

        w("DDColor Lab — Guía de uso\n\n", "h1")

        w("Descargar los pesos del proyecto\n", "h2")
        w("  Los pesos fine-tuned sobre cómics están en HuggingFace:\n\n", "dim")
        w("""\
  from huggingface_hub import hf_hub_download
  hf_hub_download(repo_id="Aleparqui/PID_proyect",
                  filename="net_g_latest.pth",
                  local_dir="./models/")
  # Renombra a: models/ddcolor_comic.pth
\n""", "code")

        w("Estructura de archivos\n", "h2")
        w("""\
  tu_proyecto/
  ├── ddcolor_desktop.py
  ├── models/
  │   ├── ddcolor_comic.pth          ← fine-tune (BasicSR checkpoint)
  │   ├── ddcolor_realistic.pth      ← modelo oficial (ModelScope / HF)
  │   └── ddcolor_artistic.pth       ← modelo artístico (ModelScope / HF)
  └── ddcolor/                       ← paquete instalado con setup.py develop
\n""", "code")

        w("Descargar los modelos oficiales\n", "h2")
        w("""\
  # Realista:
  hf_hub_download(repo_id="piddnad/DDColor-models",
                  filename="ddcolor_modelscope.pth",
                  local_dir="./models/")
  # Renombra a: models/ddcolor_realistic.pth

  # Artístico:
  hf_hub_download(repo_id="piddnad/DDColor-models",
                  filename="ddcolor_artistic.pth",
                  local_dir="./models/")
  # Renombra a: models/ddcolor_artistic.pth
\n""", "code")

        w("Instalar dependencia LPIPS\n", "h2")
        w("""\
  pip install lpips
\n""", "code")
        w("  Sin lpips instalado, la métrica LPIPS mostrará '—'.\n\n", "amber")

        w("Ground Truth\n", "h2")
        w("""\
  Al abrir una imagen, ésta se usa automáticamente como GT.
  Puedes sobreescribir con el botón "Abrir GT (opcional)"
  si dispones de una imagen de referencia distinta.
\n""", "dim")

        w("Métricas\n", "h2")
        rows = [
            ("ΔCF  ↓",        "|CF(pred) − CF(GT)|. Cercano a 0 = viveza cromática igual al GT. Con GT."),
            ("ΔE 2000  ↓",    "Error colorimétrico perceptual ΔE2000 medio píxel a píxel. Con GT."),
            ("LPIPS  ↓",      "Similitud perceptual (AlexNet). ↓ = más cercano al GT. Con GT."),
        ]
        for name, desc in rows:
            w(f"  {name:<20}", "ok")
            w(f"  {desc}\n", "dim")

        w("\nResultados esperados (cómic vs realista)\n", "h2")
        w("""\
  El modelo Cómic FT debería obtener:
    • ΔCF  cercano a 0 (viveza cromática fiel al GT de cómic)
    • ΔE   menor (colores más fieles al GT en percepción humana)
    • LPIPS menor (más similitud perceptual con el GT)

  El modelo Realista en imágenes de cómic tenderá a:
    • ΔCF  mayor (viveza cromática divergente del GT)
    • ΔE   mayor (paleta perceptualmente diferente)
    • LPIPS mayor (percepción diferente al GT)
\n""", "dim")

        txt.configure(state="disabled")

    # ═════════════════════════════════════════════════════════════════════════
    # Eventos de usuario
    # ═════════════════════════════════════════════════════════════════════════

    def _open_input(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("Todos", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"No se pudo leer: {path}")
            return
        self._input_bgr = img
        self._gt_bgr    = img
        self._lbl_gt.configure(text="Auto (misma imagen)", fg=GREEN)
        ph = bgr_to_photoimage(img, THUMB_W, THUMB_H)
        self._ph_input = ph
        self._lbl_input.configure(image=ph)
        self._lbl_input.image = ph
        self._var_status.set(f"Imagen: {Path(path).name}")

    def _open_gt(self):
        path = filedialog.askopenfilename(
            title="Seleccionar Ground Truth",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("Todos", "*.*")],
        )
        if not path:
            return
        self._gt_bgr = cv2.imread(path)
        name = Path(path).name
        self._lbl_gt.configure(
            text=name[:24] + ("…" if len(name) > 24 else ""), fg=GREEN)

    def _run(self):
        if self._input_bgr is None:
            messagebox.showwarning("DDColor Lab", "Abre una imagen primero.")
            return
        self._btn_run.configure(state="disabled")
        self._progress.start(10)
        self._var_status.set("Procesando…")
        threading.Thread(target=self._worker, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    # Worker (hilo background)
    # ─────────────────────────────────────────────────────────────────────────

    def _worker(self):
        img  = self._input_bgr
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        results = {}
        for style in MODEL_STYLES:
            if not self._model_vars[style].get():
                results[style] = None
                continue
            pipe = self._pipes.get(style)
            if pipe is None:
                results[style] = self._demo_colorize(gray, style)
            else:
                try:
                    out = pipe.process(img)
                    if out.dtype != np.uint8 and out.min() >= -1.1 and out.max() <= 1.1:
                        out = ((out + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    results[style] = out
                except Exception as e:
                    print(f"[ERROR] {style}: {e}")
                    results[style] = self._demo_colorize(gray, style)

        metrics = {
            s: (compute_metrics(results[s], self._gt_bgr) if results[s] is not None else {})
            for s in MODEL_STYLES
        }
        self.after(0, lambda: self._update(results, metrics))

    def _demo_colorize(self, gray_bgr: np.ndarray, style: str) -> np.ndarray:
        """Colorización de demostración cuando no hay modelo cargado."""
        lab = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        L = lab[:, :, 0]
        if style == "comic":
            A = np.round((np.sin(L / 25.0) * 55 + np.cos(L / 40.0) * 20) / 15) * 15
            B = np.round((np.cos(L / 30.0) * 60 - np.sin(L / 20.0) * 25) / 15) * 15
        elif style == "artistic":
            A = np.sin(L / 20.0) * 45 + np.cos(L / 35.0) * 30
            B = np.cos(L / 25.0) * 50 - np.sin(L / 15.0) * 20
        else:
            A = np.sin(L / 35.0) * 25 + np.cos(L / 55.0) * 10
            B = np.cos(L / 45.0) * 30 - np.sin(L / 30.0) * 12
        lab_out = np.stack([L, A, B], axis=2)
        return cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

    # ─────────────────────────────────────────────────────────────────────────
    # Actualización de UI tras inferencia
    # ─────────────────────────────────────────────────────────────────────────

    def _update(self, results: dict, metrics: dict):
        self._progress.stop()
        self._btn_run.configure(state="normal")
        self._results = results
        self._metrics = metrics

        # Actualizar previsualizaciones
        preview_map = {
            "comic":     (self._lbl_comic, "Desactivado"),
            "realistic": (self._lbl_real,  "Desactivado"),
            "artistic":  (self._lbl_art,   "Desactivado"),
        }
        for style, (lbl, ph_text) in preview_map.items():
            bgr = results.get(style)
            ph = (bgr_to_photoimage(bgr, PREVIEW_W, PREVIEW_H) if bgr is not None
                  else placeholder(PREVIEW_W, PREVIEW_H, ph_text))
            lbl.configure(image=ph)
            lbl.image = ph

        # Actualizar tabla de métricas
        summary_parts = self._update_metric_table(metrics)

        # Barra de estado
        mode = "DEMO" if self._pipes["comic"] is None else "REAL"
        gt_s = " · GT ✓" if self._gt_bgr is not None else ""
        self._var_status.set(f"[{mode}]{gt_s} · Completado")
        self._lbl_summary.configure(
            text="  ·  ".join(summary_parts) if summary_parts else "Sin resultados",
            fg=TEXT,
        )
        self._btn_save.configure(state="normal")

    def _update_metric_table(self, metrics: dict) -> list[str]:
        """Rellena la tabla de métricas y devuelve partes para el resumen."""
        style_keys = ("comic", "realistic", "artistic")
        summary = []

        for row_lbls, (name, key, lower, _) in zip(self._metric_rows, self.METRIC_DEFS):
            values = {s: metrics[s].get(key) for s in style_keys}
            valid  = {s: v for s, v in values.items() if v is not None}

            if len(valid) >= 2:
                best = min(valid, key=valid.get) if lower else max(valid, key=valid.get)
                for col_idx, style in enumerate(style_keys, start=1):
                    v = values[style]
                    text = ("—" if v is None
                            else (f"✓ {v:.4f}" if style == best else f"{v:.4f}"))
                    fg = GREEN if style == best and v is not None else TEXT
                    row_lbls[col_idx].configure(text=text, fg=fg)
                row_lbls[4].configure(text="↓" if lower else "↑")
                summary.append(f"{key}→{best}")
            else:
                for i in range(1, 5):
                    row_lbls[i].configure(text="—", fg=TEXT_DIM)

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # Guardar
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self):
        folder = filedialog.askdirectory(title="Carpeta de destino")
        if not folder:
            return
        name_map = {
            "comic":     "result_comic_ft.png",
            "realistic": "result_realistic_hf.png",
            "artistic":  "result_artistic_hf.png",
        }
        saved = []
        for style, fname in name_map.items():
            bgr = self._results.get(style)
            if bgr is not None:
                cv2.imwrite(str(Path(folder) / fname), bgr)
                saved.append(fname)
        if saved:
            messagebox.showinfo("DDColor Lab", "Guardado:\n" + "\n".join(saved))

    # ─────────────────────────────────────────────────────────────────────────
    # Carga de modelos en background
    # ─────────────────────────────────────────────────────────────────────────

    def _load_models_bg(self):
        def worker():
            if not TORCH_OK:
                self.after(0, lambda: (
                    self._lbl_model_info.configure(
                        text="PyTorch no instalado.\nModo DEMO activo.", fg=AMBER),
                    self._var_status.set("Modo DEMO"),
                ))
                return

            MODELS_DIR.mkdir(exist_ok=True)
            info_lines = []
            for style, fname in MODEL_FILES.items():
                pipe = try_load_pipeline(MODELS_DIR / fname)
                self._pipes[style] = pipe
                tag = fname.replace("ddcolor_", "").replace(".pth", "")
                info_lines.append(f"{'✓' if pipe else '—'} {tag}{'  (demo)' if not pipe else ''}")

            all_ok = all(p is not None for p in self._pipes.values())
            info   = "\n".join(info_lines)
            self.after(0, lambda: (
                self._lbl_model_info.configure(text=info, fg=GREEN if all_ok else AMBER),
                self._var_status.set("Listo"),
            ))

        threading.Thread(target=worker, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    App().mainloop()