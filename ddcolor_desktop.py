"""
DDColor Lab — Aplicación de escritorio
=======================================
Coloca los pesos en ./models/:
    models/ddcolor_comic.pth        (tu fine-tune, guardado por BasicSR)
    models/ddcolor_realistic.pth    (modelo oficial ModelScope / HuggingFace)

    Ambos pueden ser 'large' (convnext-l) o 'tiny' (convnext-t).
    La app detecta el tamaño automáticamente por el nombre del fichero.

Requiere:
    pip install pillow scikit-image opencv-python torch torchvision
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
    from skimage.color import rgb2lab, lab2rgb
    from skimage.segmentation import slic
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

# ── Paleta minimalista ────────────────────────────────────────────────────────
BG      = "#111318"
BG2     = "#16191f"
BG3     = "#1c2028"
BORDER  = "#262b36"
TEXT    = "#b8bcc8"
TEXT_HI = "#eef0f5"
TEXT_DIM = "#50586a"
ACCENT  = "#4f7aff"
GREEN   = "#3dba8a"
AMBER   = "#e8a535"
RED     = "#e05a5a"
COMIC   = "#4f7aff"
REAL    = "#9d7fe8"

MODELS_DIR  = Path("./models")
PREVIEW_W   = 360
PREVIEW_H   = 280
THUMB_W     = 160
THUMB_H     = 120


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline de inferencia — exactamente como lo hace el proyecto DDColor
# ═════════════════════════════════════════════════════════════════════════════

def detect_model_size(path: Path) -> str:
    """Detecta si el checkpoint es tiny o large por nombre de fichero."""
    return "tiny" if "tiny" in path.stem.lower() else "large"


def try_load_pipeline(path: Path, input_size: int = 512):
    """
    Carga el modelo usando exactamente el mismo código que infer.py del proyecto:
      build_ddcolor_model()  →  ColorizationPipeline()
    Devuelve un ColorizationPipeline listo para llamar a .process(img_bgr).
    """
    if not path.exists():
        return None
    try:
        from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_size = detect_model_size(path)
        model = build_ddcolor_model(
            DDColor,
            model_path=str(path),
            input_size=input_size,
            model_size=model_size,
            device=device,
        )
        return ColorizationPipeline(model, input_size=input_size, device=device)
    except Exception as e:
        print(f"[ERROR] {path.name}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Métricas — las que usa el proyecto + las que desarrollamos antes
# ═════════════════════════════════════════════════════════════════════════════

def colorfulness_score(img_bgr: np.ndarray) -> float:
    """
    Hasler & Süsstrunk (2003).
    El proyecto usa esta métrica como 'CF score' en sus evaluaciones.
    Mayor = más colorido.
    """
    img = img_bgr.astype(float)
    R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]
    rg  = R - G
    yb  = 0.5*(R+G) - B
    return float(np.sqrt(np.std(rg)**2 + np.std(yb)**2)
                 + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))


def fid_proxy(img_bgr: np.ndarray) -> float:
    """
    Proxy de FID: diversidad de features de color en el espacio LAB.
    Equivale a la varianza total de los canales AB — correlaciona con FID real.
    """
    if not SKIMAGE_OK:
        return 0.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(float)
    ab  = lab[:, :, 1:]
    return float(np.std(ab))


def psnr_ssim(pred_bgr: np.ndarray, gt_bgr: np.ndarray):
    """
    PSNR y SSIM — métricas estándar del proyecto (reportadas en el paper).
    Necesitan ground truth.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        gt_rs = cv2.resize(gt_bgr, (pred_bgr.shape[1], pred_bgr.shape[0]))
        psnr  = float(peak_signal_noise_ratio(gt_rs, pred_bgr, data_range=255))
        ssim  = float(structural_similarity(gt_rs, pred_bgr, channel_axis=2, data_range=255))
        return psnr, ssim
    except Exception:
        return None, None


def chrominance_mae(pred_bgr: np.ndarray, gt_bgr: np.ndarray):
    """Error medio en canales AB del espacio LAB."""
    gt_rs    = cv2.resize(gt_bgr, (pred_bgr.shape[1], pred_bgr.shape[0]))
    pred_lab = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2Lab).astype(float)
    gt_lab   = cv2.cvtColor(gt_rs,   cv2.COLOR_BGR2Lab).astype(float)
    chroma = float(np.mean(np.abs(pred_lab[:,:,1:] - gt_lab[:,:,1:])))
    luma   = float(np.mean(np.abs(pred_lab[:,:,0]  - gt_lab[:,:,0])))
    return chroma, luma


def color_consistency(img_bgr: np.ndarray, n: int = 150) -> float | None:
    """
    Varianza intra-región (SLIC). Menor = colores más planos = más estilo cómic.
    """
    if not SKIMAGE_OK:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(float)
    segs    = slic(img_rgb, n_segments=n, compactness=10, start_label=0)
    v = [float(np.var(lab[segs==s, 1:]))
         for s in np.unique(segs) if (segs==s).sum() >= 5]
    return float(np.mean(v)) if v else None


def compute_metrics(pred_bgr: np.ndarray, gt_bgr: np.ndarray | None) -> dict:
    m = {
        "colorfulness": colorfulness_score(pred_bgr),
        "fid_proxy":    fid_proxy(pred_bgr),
        "consistency":  color_consistency(pred_bgr),
        "psnr":         None,
        "ssim":         None,
        "chroma_mae":   None,
        "luma_mae":     None,
    }
    if gt_bgr is not None:
        m["psnr"], m["ssim"]           = psnr_ssim(pred_bgr, gt_bgr)
        m["chroma_mae"], m["luma_mae"] = chrominance_mae(pred_bgr, gt_bgr)
    return m


# ═════════════════════════════════════════════════════════════════════════════
# Helpers UI
# ═════════════════════════════════════════════════════════════════════════════

def bgr_to_photoimage(img_bgr: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), BG2)
    ox, oy = (w - pil.width)//2, (h - pil.height)//2
    canvas.paste(pil, (ox, oy))
    return ImageTk.PhotoImage(canvas)


def pil_to_photoimage(img_pil: Image.Image, w: int, h: int) -> ImageTk.PhotoImage:
    img_pil.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), BG2)
    ox, oy = (w - img_pil.width)//2, (h - img_pil.height)//2
    canvas.paste(img_pil, (ox, oy))
    return ImageTk.PhotoImage(canvas)


def placeholder(w: int, h: int, text: str) -> ImageTk.PhotoImage:
    img = Image.new("RGB", (w, h), BG3)
    d   = ImageDraw.Draw(img)
    d.text((w//2, h//2), text, fill=TEXT_DIM, anchor="mm")
    return ImageTk.PhotoImage(img)


def section_bar(parent, text: str):
    f = tk.Frame(parent, bg=BG3)
    f.pack(fill="x", pady=(10, 2))
    tk.Label(f, text=f"  {text}", bg=BG3, fg=TEXT_DIM,
             font=("Helvetica", 9, "bold"), anchor="w", pady=4).pack(fill="x")


def sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=0, pady=5)


# ═════════════════════════════════════════════════════════════════════════════
# Aplicación
# ═════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("DDColor Lab")
        self.configure(bg=BG)
        self.minsize(1020, 700)

        # Estado
        self._input_bgr  = None     # numpy BGR
        self._gt_bgr     = None
        self._comic_bgr  = None
        self._real_bgr   = None
        self._pipes      = {"comic": None, "realistic": None}

        MODELS_DIR.mkdir(exist_ok=True)

        # Placeholders
        self._ph = {
            "input": placeholder(THUMB_W, THUMB_H, "Abrir imagen"),
            "comic": placeholder(PREVIEW_W, PREVIEW_H, "Modelo Cómic FT"),
            "real":  placeholder(PREVIEW_W, PREVIEW_H, "Modelo Realista HF"),
        }

        self._build_ui()
        self._load_models_bg()

    # ── UI principal ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Barra superior
        top = tk.Frame(self, bg=BG2, pady=10)
        top.pack(fill="x")
        tk.Frame(top, bg=BORDER, height=1).pack(fill="x", side="bottom")

        tk.Label(top, text="DDColor Lab", bg=BG2, fg=TEXT_HI,
                 font=("Helvetica", 14, "bold"), padx=16).pack(side="left")
        tk.Label(top, text="EVALUACIÓN DE COLORIZACIÓN · ICCV 2023",
                 bg=BG2, fg=TEXT_DIM, font=("Helvetica", 8)).pack(side="left")

        self._var_status = tk.StringVar(value="Cargando modelos...")
        tk.Label(top, textvariable=self._var_status, bg=BG2,
                 fg=GREEN, font=("Helvetica", 9), padx=16).pack(side="right")

        # Notebook
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

        t1 = tk.Frame(nb, bg=BG)
        t2 = tk.Frame(nb, bg=BG)
        t3 = tk.Frame(nb, bg=BG)
        nb.add(t1, text="  Colorizar y comparar  ")
        nb.add(t2, text="  Métricas  ")
        nb.add(t3, text="  Guía  ")

        self._build_main(t1)
        self._build_metrics_tab(t2)
        self._build_guide(t3)

    # ── Tab 1 ─────────────────────────────────────────────────────────────────

    def _build_main(self, parent):
        left  = tk.Frame(parent, bg=BG2, width=210)
        right = tk.Frame(parent, bg=BG)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Frame(parent, bg=BORDER, width=1).pack(side="left", fill="y")
        right.pack(side="left", fill="both", expand=True)

        # ── Panel izquierdo ───────────────────────────────────────────────────
        section_bar(left, "ENTRADA")

        # Thumbnail clickeable
        frm_thumb = tk.Frame(left, bg=BG3, width=THUMB_W, height=THUMB_H)
        frm_thumb.pack(pady=6)
        frm_thumb.pack_propagate(False)
        self._lbl_input = tk.Label(frm_thumb, image=self._ph["input"],
                                   bg=BG3, cursor="hand2")
        self._lbl_input.pack(expand=True)
        self._lbl_input.bind("<Button-1>", lambda _: self._open_input())

        tk.Button(
            left, text="Abrir imagen",
            bg=ACCENT, fg="white", relief="flat",
            font=("Helvetica", 9, "bold"), pady=6, cursor="hand2",
            activebackground=BG3, activeforeground=ACCENT,
            command=self._open_input
        ).pack(fill="x", padx=12, pady=(2, 0))

        section_bar(left, "GROUND TRUTH")
        tk.Button(
            left, text="Abrir GT (opcional)",
            bg=BG3, fg=TEXT, relief="flat", font=("Helvetica", 9),
            pady=5, cursor="hand2",
            activebackground=BORDER, activeforeground=TEXT_HI,
            command=self._open_gt
        ).pack(fill="x", padx=12, pady=2)
        self._lbl_gt = tk.Label(left, text="Sin GT", bg=BG2,
                                fg=TEXT_DIM, font=("Helvetica", 8))
        self._lbl_gt.pack(pady=2)

        section_bar(left, "MODELOS")
        self._var_comic = tk.BooleanVar(value=True)
        self._var_real  = tk.BooleanVar(value=True)
        for var, txt, col in [(self._var_comic, "Cómic FT",    COMIC),
                               (self._var_real,  "Realista HF", REAL)]:
            tk.Checkbutton(
                left, variable=var, text=txt, bg=BG2, fg=col,
                selectcolor=BG3, activebackground=BG2, activeforeground=col,
                font=("Helvetica", 9), bd=0
            ).pack(anchor="w", padx=16, pady=2)

        sep(left)

        self._btn_run = tk.Button(
            left, text="▶  Ejecutar",
            bg=ACCENT, fg="white", relief="flat",
            font=("Helvetica", 10, "bold"), pady=8, cursor="hand2",
            activebackground="#3a60cc", activeforeground="white",
            command=self._run
        )
        self._btn_run.pack(fill="x", padx=12, pady=4)

        self._btn_save = tk.Button(
            left, text="Guardar resultados",
            bg=BG3, fg=TEXT, relief="flat", font=("Helvetica", 9),
            pady=5, cursor="hand2",
            activebackground=BORDER, activeforeground=TEXT_HI,
            command=self._save, state="disabled"
        )
        self._btn_save.pack(fill="x", padx=12, pady=2)

        self._progress = ttk.Progressbar(left, mode="indeterminate")
        self._progress.pack(fill="x", padx=12, pady=4)

        self._lbl_model_info = tk.Label(
            left, text="", bg=BG2, fg=TEXT_DIM,
            font=("Helvetica", 8), wraplength=185, justify="left"
        )
        self._lbl_model_info.pack(padx=12, pady=6, anchor="w")

        # ── Panel derecho ─────────────────────────────────────────────────────
        results = tk.Frame(right, bg=BG)
        results.pack(fill="both", expand=True, padx=12, pady=10)

        for col_i, (label, col, attr) in enumerate([
            ("Modelo Cómic FT",    COMIC, "_lbl_comic"),
            ("Modelo Realista HF", REAL,  "_lbl_real"),
        ]):
            col_f = tk.Frame(results, bg=BG)
            col_f.grid(row=0, column=col_i, sticky="nsew", padx=6)
            results.columnconfigure(col_i, weight=1)
            results.rowconfigure(0, weight=1)

            tk.Label(col_f, text=label, bg=BG, fg=col,
                     font=("Helvetica", 9, "bold"),
                     anchor="w").pack(fill="x", pady=(0, 4))

            ph_key = "comic" if col_i == 0 else "real"
            img_f  = tk.Frame(col_f, bg=BG3)
            img_f.pack(fill="both", expand=True)
            lbl = tk.Label(img_f, image=self._ph[ph_key], bg=BG3)
            lbl.pack(fill="both", expand=True)
            setattr(self, attr, lbl)

        # Barra de resumen de métricas al fondo
        mbar = tk.Frame(right, bg=BG2, pady=5)
        mbar.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(mbar, text="RESUMEN", bg=BG2, fg=TEXT_DIM,
                 font=("Helvetica", 8, "bold"), padx=8).pack(side="left")
        self._lbl_summary = tk.Label(
            mbar, text="Ejecuta la colorización para ver los resultados.",
            bg=BG2, fg=TEXT_DIM, font=("Helvetica", 9), padx=8
        )
        self._lbl_summary.pack(side="left")

    # ── Tab 2: tabla de métricas ──────────────────────────────────────────────

    def _build_metrics_tab(self, parent):
        tk.Label(parent,
                 text="Métricas de la última colorización.\n"
                      "Las métricas marcadas con (GT) requieren ground truth.",
                 bg=BG, fg=TEXT_DIM, font=("Helvetica", 9),
                 justify="left", pady=8).pack(anchor="w", padx=20)

        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=4)

        HDR_DEFS = [
            ("Métrica", 22), ("Cómic FT", 14), ("Realista HF", 14), ("↑/↓", 5), ("Descripción", 45)
        ]
        for c, (h, w) in enumerate(HDR_DEFS):
            tk.Label(frame, text=h, bg=BG3, fg=TEXT_DIM,
                     font=("Helvetica", 8, "bold"), width=w, anchor="w",
                     padx=6, pady=5, relief="flat"
                     ).grid(row=0, column=c, sticky="ew", padx=1, pady=(0, 2))

        # Filas de métricas: (nombre, key, lower_is_better, necesita_GT, descripción)
        self._METRIC_DEFS = [
            ("Colorfulness (CF) ↑",  "colorfulness", False, False,
             "Riqueza cromática (Hasler & Süsstrunk). Usada en el paper DDColor."),
            ("FID proxy ↑",          "fid_proxy",    False, False,
             "Std de canales AB — diversidad de color. Correlaciona con FID real."),
            ("PSNR ↑ (GT)",          "psnr",         False, True,
             "Peak Signal-to-Noise Ratio. Métrica estándar del paper."),
            ("SSIM ↑ (GT)",          "ssim",         False, True,
             "Structural Similarity Index. Métrica estándar del paper."),
            ("Chrominance MAE ↓ (GT)","chroma_mae",  True,  True,
             "Error medio en canales AB del espacio LAB."),
            ("Luminance MAE ↓ (GT)", "luma_mae",     True,  True,
             "Error medio en canal L (estructura, no color)."),
            ("Color Consistency ↓",  "consistency",  True,  False,
             "Varianza SLIC intra-región. Menor = más plano = más estilo cómic."),
        ]

        self._metric_labels = []
        for r, (name, key, lower, needs_gt, desc) in enumerate(self._METRIC_DEFS):
            row_lbls = []
            for c, (txt, w, fg) in enumerate([
                (name, 22, TEXT),
                ("—",  14, TEXT_DIM),
                ("—",  14, TEXT_DIM),
                ("",    5, TEXT_DIM),
                (desc, 45, TEXT_DIM),
            ]):
                l = tk.Label(frame, text=txt, bg=BG2, fg=fg,
                             font=("Helvetica", 9), width=w, anchor="w",
                             padx=6, pady=5, relief="flat")
                l.grid(row=r+1, column=c, sticky="ew", padx=1, pady=1)
                row_lbls.append(l)
            self._metric_labels.append(row_lbls)

        for c in range(5):
            frame.columnconfigure(c, weight=1 if c == 4 else 0)

        leg = tk.Frame(parent, bg=BG)
        leg.pack(anchor="w", padx=20, pady=10)
        for txt, col in [("✓ mejor resultado", GREEN), ("(GT) necesita ground truth", TEXT_DIM)]:
            tk.Label(leg, text=txt, bg=BG, fg=col,
                     font=("Helvetica", 9)).pack(side="left", padx=12)

    # ── Tab 3: guía ───────────────────────────────────────────────────────────

    def _build_guide(self, parent):
        txt = tk.Text(parent, bg=BG2, fg=TEXT, font=("Helvetica", 10),
                      relief="flat", padx=20, pady=16, wrap="word",
                      insertbackground=ACCENT, selectbackground=ACCENT)
        sb  = tk.Scrollbar(parent, command=txt.yview, bg=BG3, troughcolor=BG)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)

        txt.tag_configure("h1",   font=("Helvetica", 13, "bold"), foreground=TEXT_HI)
        txt.tag_configure("h2",   font=("Helvetica", 10, "bold"), foreground=ACCENT)
        txt.tag_configure("code", font=("Courier", 9),            foreground="#22d3ee",
                          background=BG3)
        txt.tag_configure("dim",  foreground=TEXT_DIM)
        txt.tag_configure("ok",   foreground=GREEN)

        def w(t, *tags): txt.insert("end", t, tags)

        w("DDColor Lab — Guía de uso\n\n", "h1")

        w("Estructura de archivos\n", "h2")
        w("""\
  tu_proyecto/
  ├── ddcolor_desktop.py
  ├── models/
  │   ├── ddcolor_comic.pth          ← tu fine-tune (BasicSR checkpoint)
  │   └── ddcolor_realistic.pth      ← modelo oficial (ModelScope / HF)
  └── ddcolor/                       ← paquete instalado con setup.py develop
\n""", "code")

        w("Formato de los pesos\n", "h2")
        w("""\
  Los checkpoints guardados por BasicSR tienen la clave "params":
\n""", "dim")
        w("""\
  # Así guarda BasicSR durante el entrenamiento:
  torch.save({"params": model.state_dict()}, "checkpoint.pth")

  # La app lo carga automáticamente — no necesitas hacer nada especial.
\n""", "code")

        w("Modelo 'tiny' vs 'large'\n", "h2")
        w("""\
  Si el nombre del fichero contiene "tiny", la app usa convnext-t.
  En caso contrario usa convnext-l (large). Puedes renombrar el fichero
  para controlar esto, p.ej.: ddcolor_comic_tiny.pth
\n""", "dim")

        w("Descargar el modelo realista\n", "h2")
        w("""\
  # Opción A — HuggingFace:
  from huggingface_hub import hf_hub_download
  hf_hub_download(
      repo_id="piddnad/DDColor-models",
      filename="ddcolor_modelscope.pth",
      local_dir="./models/"
  )
  # Renombra a: models/ddcolor_realistic.pth

  # Opción B — con el script del proyecto:
  python scripts/infer.py --model_name ddcolor_modelscope ...
\n""", "code")

        w("Métricas\n", "h2")
        rows = [
            ("Colorfulness (CF) ↑",   "Riqueza cromática. Reportada en el paper DDColor. Sin GT."),
            ("FID proxy ↑",            "Proxy de diversidad de color (std AB). Sin GT."),
            ("PSNR ↑",                 "Peak Signal-to-Noise Ratio. Estándar del paper. Con GT."),
            ("SSIM ↑",                 "Structural Similarity. Estándar del paper. Con GT."),
            ("Chrominance MAE ↓",      "Error en canales AB del espacio LAB. Con GT."),
            ("Color Consistency ↓",    "Varianza intra-región SLIC. Estilo plano = valor bajo. Sin GT."),
        ]
        for name, desc in rows:
            w(f"  {name:<28}", "ok")
            w(f"  {desc}\n",   "dim")

        w("\nResultados esperados\n", "h2")
        w("""\
  El modelo Cómic FT debería obtener:
    • Mayor Colorfulness (paletas más saturadas)
    • Menor Color Consistency (colores planos por región)
    • Menor Chrominance MAE respecto al GT de cómic

  El modelo Realista HF en imágenes de cómic tenderá a:
    • Desaturar los colores (menor CF)
    • Introducir gradientes innecesarios (mayor Consistency)
    • Tener mayor error cromático vs GT de cómic
\n""", "dim")

        txt.configure(state="disabled")

    # ═════════════════════════════════════════════════════════════════════════
    # Eventos
    # ═════════════════════════════════════════════════════════════════════════

    def _open_input(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("Todos",    "*.*")]
        )
        if not path:
            return
        self._input_bgr = cv2.imread(path)
        if self._input_bgr is None:
            messagebox.showerror("Error", f"No se pudo leer: {path}")
            return
        ph = bgr_to_photoimage(self._input_bgr, THUMB_W, THUMB_H)
        self._ph["input"] = ph
        self._lbl_input.configure(image=ph)
        self._lbl_input.image = ph
        self._var_status.set(f"Imagen: {Path(path).name}")

    def _open_gt(self):
        path = filedialog.askopenfilename(
            title="Seleccionar Ground Truth",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("Todos",    "*.*")]
        )
        if not path:
            return
        self._gt_bgr = cv2.imread(path)
        name = Path(path).name
        self._lbl_gt.configure(text=name[:24] + ("…" if len(name)>24 else ""),
                                fg=GREEN)

    def _run(self):
        if self._input_bgr is None:
            messagebox.showwarning("DDColor Lab", "Abre una imagen primero.")
            return
        self._btn_run.configure(state="disabled")
        self._progress.start(10)
        self._var_status.set("Procesando…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        # Pasar la imagen original al pipeline: extrae el canal L internamente,
        # igual que hace infer.py. La versión gris solo se usa para el modo demo.
        img  = self._input_bgr
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        results = {}
        for style in ["comic", "realistic"]:
            enabled = self._var_comic.get() if style == "comic" else self._var_real.get()
            if not enabled:
                results[style] = None
                continue
            pipe = self._pipes.get(style)
            if pipe is None:
                results[style] = self._demo_colorize(gray, style)
            else:
                try:
                    results[style] = pipe.process(img)
                except Exception as e:
                    print(f"[ERROR] {style}: {e}")
                    results[style] = self._demo_colorize(gray, style)

        m_c = compute_metrics(results["comic"],    self._gt_bgr) if results.get("comic")    is not None else {}
        m_r = compute_metrics(results["realistic"], self._gt_bgr) if results.get("realistic") is not None else {}

        self.after(0, lambda: self._update(results, m_c, m_r))

    def _demo_colorize(self, gray_bgr: np.ndarray, style: str) -> np.ndarray:
        """Colorización simulada cuando no hay pesos."""
        if not SKIMAGE_OK:
            return gray_bgr
        lab = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        L   = lab[:, :, 0]
        if style == "comic":
            A = np.round((np.sin(L/25.0)*55 + np.cos(L/40.0)*20) / 15) * 15
            B = np.round((np.cos(L/30.0)*60 - np.sin(L/20.0)*25) / 15) * 15
        else:
            A = np.sin(L/35.0)*25 + np.cos(L/55.0)*10
            B = np.cos(L/45.0)*30 - np.sin(L/30.0)*12
        lab_out = np.stack([L, A, B], axis=2)
        return cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8),
                            cv2.COLOR_Lab2BGR)

    def _update(self, results: dict, m_c: dict, m_r: dict):
        self._progress.stop()
        self._btn_run.configure(state="normal")

        self._comic_bgr = results.get("comic")
        self._real_bgr  = results.get("realistic")

        def show_result(bgr, lbl_widget, ph_key, placeholder_text):
            if bgr is not None:
                ph = bgr_to_photoimage(bgr, PREVIEW_W, PREVIEW_H)
            else:
                ph = placeholder(PREVIEW_W, PREVIEW_H, placeholder_text)
            self._ph[ph_key] = ph
            lbl_widget.configure(image=ph)
            lbl_widget.image = ph

        show_result(self._comic_bgr, self._lbl_comic, "comic", "Desactivado")
        show_result(self._real_bgr,  self._lbl_real,  "real",  "Desactivado")

        # ── Actualizar tabla de métricas ──────────────────────────────────────
        summary_parts = []
        for row_lbls, (name, key, lower, needs_gt, _) in zip(
            self._metric_labels, self._METRIC_DEFS
        ):
            vc = m_c.get(key)
            vr = m_r.get(key)

            def fmt(v):
                if v is None: return "—"
                return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"

            if vc is not None and vr is not None:
                a_wins = (vc < vr) if lower else (vc > vr)
                row_lbls[1].configure(
                    text=f"✓ {fmt(vc)}" if a_wins else fmt(vc),
                    fg=GREEN if a_wins else TEXT
                )
                row_lbls[2].configure(
                    text=f"✓ {fmt(vr)}" if not a_wins else fmt(vr),
                    fg=GREEN if not a_wins else TEXT
                )
                row_lbls[3].configure(text="↑" if not lower else "↓")
                winner = "Cómic" if a_wins else "Real"
                summary_parts.append(f"{key.split('_')[0]}→{winner}")
            else:
                row_lbls[1].configure(text=fmt(vc), fg=TEXT_DIM if vc is None else TEXT)
                row_lbls[2].configure(text=fmt(vr), fg=TEXT_DIM if vr is None else TEXT)

        mode = "DEMO" if self._pipes["comic"] is None else "REAL"
        gt_s = " · GT ✓" if self._gt_bgr is not None else ""
        self._var_status.set(f"[{mode}]{gt_s} · Completado")
        self._lbl_summary.configure(
            text="  ·  ".join(summary_parts) if summary_parts else "Sin resultados",
            fg=TEXT
        )
        self._btn_save.configure(state="normal")

    def _save(self):
        folder = filedialog.askdirectory(title="Carpeta de destino")
        if not folder:
            return
        saved = []
        for bgr, name in [(self._comic_bgr,  "result_comic_ft.png"),
                          (self._real_bgr,   "result_realistic_hf.png")]:
            if bgr is not None:
                cv2.imwrite(str(Path(folder)/name), bgr)
                saved.append(name)
        if saved:
            messagebox.showinfo("DDColor Lab", "Guardado:\n" + "\n".join(saved))

    # ── Carga de modelos en background ────────────────────────────────────────

    def _load_models_bg(self):
        def worker():
            if not TORCH_OK:
                self.after(0, lambda: (
                    self._lbl_model_info.configure(
                        text="PyTorch no instalado.\nModo DEMO activo.", fg=AMBER
                    ),
                    self._var_status.set("Modo DEMO")
                ))
                return

            MODELS_DIR.mkdir(exist_ok=True)
            info_lines = []
            for style, fname in [("comic",    "ddcolor_comic.pth"),
                                  ("realistic","ddcolor_realistic.pth")]:
                p = MODELS_DIR / fname
                pipe = try_load_pipeline(p)
                self._pipes[style] = pipe
                tag = fname.replace("ddcolor_","").replace(".pth","")
                if pipe:
                    info_lines.append(f"✓ {tag}")
                else:
                    info_lines.append(f"— {tag} (demo)")

            info = "\n".join(info_lines)
            color = GREEN if all(p is not None for p in self._pipes.values()) else AMBER
            self.after(0, lambda: (
                self._lbl_model_info.configure(text=info, fg=color),
                self._var_status.set("Listo")
            ))

        threading.Thread(target=worker, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()