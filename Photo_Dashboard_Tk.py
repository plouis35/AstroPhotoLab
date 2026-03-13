"""
Photo_Dashboard_Tk.py
Application autonome de retouche photo astronomique — version Tkinter.
Fonctionnellement identique à la version PyQt5.

Dépendances :
    pip install matplotlib numpy astropy photutils scipy pillow imageio scikit-image
    (tkinter est inclus dans Python — sous Termux : pkg install python-tkinter)

Usage :
    python Photo_Dashboard_Tk.py
"""

import os, sys, time
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from astropy.visualization import AsinhStretch
import tkinter as tk
from tkinter import ttk, filedialog, font as tkfont

from Photo_Lab import PhotoLab

# ─────────────────────────────────────────────────────────────────────────────
# ÉCHELLE — ajustez cette valeur selon votre écran
#   1.0 = taille normale (PC/Mac)
#   1.8 = tablette Android (recommandé Redmi Pad)
#   2.0 = très grande tablette ou affichage HiDPI
# ─────────────────────────────────────────────────────────────────────────────

SCALE = 1.8

# Tailles dérivées
FONT_SM   = ('TkDefaultFont', round(9  * SCALE))
FONT_MD   = ('TkDefaultFont', round(10 * SCALE))
FONT_MONO = ('Courier',       round(9  * SCALE))
FONT_DOT  = ('TkDefaultFont', round(14 * SCALE))
BTN_PAD_X = round(8  * SCALE)
BTN_PAD_Y = round(5  * SCALE)
SL_LEN    = round(14 * SCALE)   # longueur poignée slider
SL_WIDTH  = round(8  * SCALE)   # épaisseur rainure slider
LBL_W     = round(10 * SCALE)   # largeur label slider (en caractères)
VAL_W     = round(6  * SCALE)   # largeur valeur slider (en caractères)
LEFT_W    = round(300 * SCALE)  # largeur panneau gauche

BG       = '#1a1a2e'
BG2      = '#12122a'
BG3      = '#2d2d44'
ACCENT   = '#00bcd4'
FG       = '#e0e0e0'
FG_DIM   = '#888888'
BTN_LOAD = '#1565c0'
BTN_EXP  = '#1b5e20'
RED      = '#e74c3c'
GREEN    = '#2ecc71'
BORDER   = '#444444'


# ─────────────────────────────────────────────────────────────────────────────
# WIDGET : SLIDER AVEC LABEL + VALEUR
# ─────────────────────────────────────────────────────────────────────────────

class LabeledSlider(tk.Frame):
    """Slider flottant : [label][slider][valeur]
    Le callback n'est déclenché qu'au relâchement (ButtonRelease)."""

    def __init__(self, parent, label, vmin, vmax, vstep, vdefault,
                 callback=None, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._min      = vmin
        self._max      = vmax
        self._step     = vstep
        self._steps    = round((vmax - vmin) / vstep)
        self._callback = callback

        # Label
        lbl = tk.Label(self, text=label, width=LBL_W, anchor='w',
                       bg=BG, fg=FG_DIM, font=FONT_SM)
        lbl.pack(side='left')

        # Slider
        self._scale = tk.Scale(
            self, from_=0, to=self._steps,
            orient='horizontal', showvalue=False,
            bg=BG, fg=FG, troughcolor=BG3, highlightthickness=0,
            bd=0, sliderlength=SL_LEN, width=SL_WIDTH,
            command=self._on_drag,
        )
        self._scale.set(self._val_to_int(vdefault))
        self._scale.pack(side='left', fill='x', expand=True)
        self._scale.bind('<ButtonRelease-1>', self._on_release)

        # Valeur
        self._val_lbl = tk.Label(self, text=f'{vdefault:.3g}', width=VAL_W,
                                  anchor='e', bg=BG, fg=FG, font=FONT_MONO)
        self._val_lbl.pack(side='left')

    def _val_to_int(self, v):
        return round((v - self._min) / self._step)

    def _val_to_frac(self, v):
        return v

    def _int_to_val(self, i):
        return self._min + int(i) * self._step

    def _on_drag(self, i):
        v = self._int_to_val(i)
        self._val_lbl.config(text=f'{v:.3g}')

    def _on_release(self, event=None):
        if self._callback:
            self._callback(self.value)

    @property
    def value(self):
        return self._int_to_val(self._scale.get())

    def set_value(self, v):
        self._scale.set(self._val_to_int(v))
        self._val_lbl.config(text=f'{v:.3g}')


# ─────────────────────────────────────────────────────────────────────────────
# TOOLBAR MATPLOTLIB ÉPURÉE
# ─────────────────────────────────────────────────────────────────────────────

class AstroToolbar(NavigationToolbar2Tk):
    """Toolbar sans Save ni Subplots."""
    toolitems = [t for t in NavigationToolbar2Tk.toolitems
                 if t[0] in ('Home', 'Back', 'Forward', None, 'Pan', 'Zoom')]


# ─────────────────────────────────────────────────────────────────────────────
# FENÊTRE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

class PhotoDashboardTk(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Photo Lab — Astronomical Image Processing")
        self.geometry("1400x860")
        self.configure(bg=BG)

        # Backend
        self.lab = PhotoLab()
        self.lab.log = self._log

        self._current_dir = os.getcwd()
        self._worker      = None
        self._pending     = None   # (update_masks, log_msg)
        self._after_id    = None   # debounce

        self._build_ui()
        plt.style.use('dark_background')

    # ── CONSTRUCTION UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Séparateur principal : gauche (300px fixe) | droite (expand)
        paned = tk.PanedWindow(self, orient='horizontal', bg=BG,
                               sashwidth=4, sashrelief='flat')
        paned.pack(fill='both', expand=True)

        # ── PANNEAU GAUCHE ────────────────────────────────────────────────────
        left_outer = tk.Frame(paned, bg=BG, width=LEFT_W)
        left_outer.pack_propagate(False)

        canvas_scroll = tk.Canvas(left_outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_outer, orient='vertical',
                                  command=canvas_scroll.yview)
        self._left_frame = tk.Frame(canvas_scroll, bg=BG)

        self._left_frame.bind('<Configure>',
            lambda e: canvas_scroll.configure(
                scrollregion=canvas_scroll.bbox('all')))

        canvas_scroll.create_window((0, 0), window=self._left_frame, anchor='nw')
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        paned.add(left_outer, minsize=round(280*SCALE))

        # Bind scroll molette
        self._left_frame.bind_all('<MouseWheel>',
            lambda e: canvas_scroll.yview_scroll(-1*(e.delta//120), 'units'))

        self._build_left()

        # ── PANNEAU DROIT ─────────────────────────────────────────────────────
        right_frame = tk.Frame(paned, bg=BG)
        paned.add(right_frame, stretch='always')
        self._build_right(right_frame)

    def _group(self, parent, title):
        """Crée un LabelFrame stylé comme un QGroupBox."""
        f = tk.LabelFrame(parent, text=title, bg=BG, fg=ACCENT,
                          font=FONT_MD,
                          bd=1, relief='solid',
                          labelanchor='nw', padx=4, pady=4)
        f.pack(fill='x', padx=4, pady=3)
        return f

    def _build_left(self):
        p = self._left_frame

        # ── FILES ─────────────────────────────────────────────────────────────
        grp = self._group(p, 'FILES')

        self._files_lbl = tk.Label(grp, text='No file loaded', bg=BG,
                                   fg=FG_DIM, font=FONT_SM,
                                   wraplength=round(260*SCALE), justify='left')
        self._files_lbl.pack(fill='x', pady=2)

        btn_row = tk.Frame(grp, bg=BG)
        btn_row.pack(fill='x', pady=2)
        tk.Button(btn_row, text='⬇  Load…', bg=BTN_LOAD, fg=FG,
                  activebackground='#1976d2', relief='flat', bd=0,
                  padx=BTN_PAD_X, pady=BTN_PAD_Y, cursor='hand2',
                  font=FONT_MD,
                  command=self._on_load_click).pack(side='left', fill='x', expand=True, padx=(0,2))
        tk.Button(btn_row, text='💾  Export', bg=BTN_EXP, fg=FG,
                  activebackground='#2e7d32', relief='flat', bd=0,
                  padx=BTN_PAD_X, pady=BTN_PAD_Y, cursor='hand2',
                  font=FONT_MD,
                  command=self._export).pack(side='left', fill='x', expand=True)

        # ── DISPLAY ───────────────────────────────────────────────────────────
        grp2 = self._group(p, 'DISPLAY')
        chk_row = tk.Frame(grp2, bg=BG)
        chk_row.pack(fill='x')

        self._var_grad  = tk.BooleanVar(value=False)
        self._var_masks = tk.BooleanVar(value=False)

        tk.Checkbutton(chk_row, text='Gradient removal',
                       variable=self._var_grad, bg=BG, fg=FG,
                       selectcolor=BG3, activebackground=BG,
                       font=FONT_SM,
                       command=lambda: self._refresh(False, 'Gradient removal')).pack(side='left')
        tk.Checkbutton(chk_row, text='Show masks',
                       variable=self._var_masks, bg=BG, fg=FG,
                       selectcolor=BG3, activebackground=BG,
                       font=FONT_SM,
                       command=lambda: self._refresh(False, 'Show masks')).pack(side='left', padx=8)

        # ── HISTOGRAM ─────────────────────────────────────────────────────────
        grp3 = self._group(p, 'HISTOGRAM  (red = black point)')

        self._fig_h = Figure(figsize=(2.8, 1.5), facecolor='black')
        self._fig_h.subplots_adjust(0, 0, 1, 1)
        self._ax_h = self._fig_h.add_subplot(111)
        self._ax_h.axis('off')
        self._canvas_h = FigureCanvasTkAgg(self._fig_h, master=grp3)
        self._canvas_h.get_tk_widget().pack(fill='x')

        # Sliders Min/Max
        self._sl_level_min = LabeledSlider(grp3, 'Min', 0.0, 0.5, 0.001, 0.0,
                                            callback=lambda v: self._refresh(False, 'Level Min'))
        self._sl_level_min.pack(fill='x', pady=1)
        self._sl_level_max = LabeledSlider(grp3, 'Max', 0.5, 1.0, 0.001, 1.0,
                                            callback=lambda v: self._refresh(False, 'Level Max'))
        self._sl_level_max.pack(fill='x', pady=1)

        # ── SLIDERS ───────────────────────────────────────────────────────────
        slider_defs = [
            ('COLOR BALANCE', [
                ('Red',        0.5, 2.0, 0.05, 1.0),
                ('Green',      0.5, 2.0, 0.05, 1.0),
                ('Blue',       0.5, 2.0, 0.05, 1.0),
            ]),
            ('MASKS', [
                ('Stars σ',    0.0, 20.0, 0.5,  0.0),
                ('Galaxy σ',   0.0, 80.0, 0.5,  0.0),
            ]),
            ('COSMETICS', [
                ('Stretch',    0.01, 0.1,  0.001, 0.01),
                ('BlackPt',    0.0,  2.0,  0.01,  1.0),
                ('Clarity',    0.0,  2.0,  0.1,   0.0),
                ('Denoise',    0.0,  2.0,  0.1,   0.0),
                ('Saturation', 0.0,  3.0,  0.1,   0.0),
            ]),
        ]

        self.sliders = {}
        for grp_name, items in slider_defs:
            grp = self._group(p, grp_name)
            for name, vmin, vmax, vstep, vdef in items:
                is_mask = name in ['Stars σ', 'Galaxy σ']
                s = LabeledSlider(grp, name, vmin, vmax, vstep, vdef,
                                  callback=lambda v, n=name, m=is_mask:
                                      self._refresh(m, f'Setting: {n}'))
                s.pack(fill='x', pady=1)
                self.sliders[name] = s

    def _build_right(self, parent):
        # Barre de statut
        status_bar = tk.Frame(parent, bg=BG)
        status_bar.pack(fill='x', padx=4, pady=2)

        self._status_dot = tk.Label(status_bar, text='●', fg=GREEN,
                                     bg=BG, font=FONT_DOT)
        self._status_dot.pack(side='left')
        self._log_lbl = tk.Label(status_bar, text='Ready', fg=FG_DIM,
                                  bg=BG, font=FONT_SM)
        self._log_lbl.pack(side='left', padx=4)

        # Canvas principal matplotlib
        self._fig = Figure(figsize=(9, 7), facecolor='black')
        self._fig.subplots_adjust(0, 0, 1, 1)
        self._ax = self._fig.add_subplot(111)
        self._ax.axis('off')

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        toolbar = AstroToolbar(self._canvas, parent, pack_toolbar=False)
        toolbar.config(background=BG3)
        for btn in toolbar.winfo_children():
            try: btn.config(background=BG3, foreground=FG)
            except: pass
        toolbar.pack(fill='x', padx=4)
        self._canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

    # ── LOGGER ────────────────────────────────────────────────────────────────

    def _log(self, msg):
        # thread-safe via after()
        self.after(0, lambda: self._log_lbl.config(text=str(msg)))

    def _set_status(self, busy: bool):
        color = RED if busy else GREEN
        self.after(0, lambda: self._status_dot.config(fg=color))

    # ── CHARGEMENT ────────────────────────────────────────────────────────────

    def _on_load_click(self):
        paths = filedialog.askopenfilenames(
            title='Select image files',
            initialdir=self._current_dir,
            filetypes=[('Images', '*.png *.tif *.tiff *.fit *.fits'),
                       ('All files', '*.*')]
        )
        if not paths:
            return
        self._current_dir = os.path.dirname(paths[0])
        names = [os.path.basename(p) for p in paths]
        self._files_lbl.config(
            text=f"{len(paths)} file(s) : {', '.join(names)}", fg=FG)

        self._set_status(True)
        self._log(f'Loading {len(paths)} file(s)…')

        def _do():
            self.lab.load_files(list(paths))
            self.after(0, self._on_load_done)

        self._worker = threading.Thread(target=_do, daemon=True)
        self._worker.start()

    def _on_load_done(self):
        self._refresh(True, 'Refreshing image…')

    # ── REFRESH (debounce) ────────────────────────────────────────────────────

    def _refresh(self, update_masks=False, log_msg=None):
        if self.lab.calibrated is None:
            return
        prev = self._pending
        self._pending = (
            update_masks or (prev[0] if prev else False),
            log_msg or (prev[1] if prev else None)
        )
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(300, self._do_pending_refresh)

    def _do_pending_refresh(self):
        if self._pending is None:
            return
        if self._worker and self._worker.is_alive():
            self._after_id = self.after(200, self._do_pending_refresh)
            return

        update_masks, log_msg = self._pending
        self._pending = None

        if log_msg:
            self._log(log_msg)
        self._set_status(True)

        # Capture des valeurs dans le thread UI
        params = dict(
            update_masks = update_masks,
            show_masks   = self._var_masks.get(),
            stretch      = self.sliders['Stretch'].value,
            bp_ratio     = self.sliders['BlackPt'].value,
            clarity      = self.sliders['Clarity'].value,
            denoise      = self.sliders['Denoise'].value,
            saturation   = self.sliders['Saturation'].value,
            galaxy_sigma = self.sliders['Galaxy σ'].value,
            stars_sigma  = self.sliders['Stars σ'].value,
            rgb          = (self.sliders['Red'].value,
                            self.sliders['Green'].value,
                            self.sliders['Blue'].value),
            do_grad      = self._var_grad.get(),
            level_min    = self._sl_level_min.value,
            level_max    = self._sl_level_max.value,
        )

        def _do():
            try:
                if params['update_masks']:
                    self.lab.extract_masks(
                        self.lab.calibrated,
                        params['stars_sigma'],
                        params['galaxy_sigma']
                    )

                ax = self._ax
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                has_zoom = xlim not in [(0.0, 1.0), (0, 1)]
                ax.clear()

                if params['show_masks'] and self.lab.masks:
                    m = np.zeros((*self.lab.calibrated.shape[:2], 3))
                    m += self.lab.masks['stars'][:, :, None] * [1, 0, 0]
                    if params['galaxy_sigma'] > 0:
                        m += self.lab.masks['galaxy'][:, :, None] * [0, 1, 0]
                        m += self.lab.masks['background'][:, :, None] * [0, 0, 0.2]
                    ax.imshow(m)
                else:
                    img = self.lab.apply_cosmetics(
                        stretch      = params['stretch'],
                        bp_ratio     = params['bp_ratio'],
                        clarity      = params['clarity'],
                        denoise      = params['denoise'],
                        saturation   = params['saturation'],
                        galaxy_sigma = params['galaxy_sigma'],
                        stars_sigma  = params['stars_sigma'],
                        rgb          = params['rgb'],
                        do_grad      = params['do_grad'],
                    )
                    lmin, lmax = params['level_min'], params['level_max']
                    if lmin > 0.0 or lmax < 1.0:
                        img = np.clip((img - lmin) / max(lmax - lmin, 1e-6), 0, 1)
                    ax.imshow(np.squeeze(img),
                              cmap='gray' if img.ndim == 2 or img.shape[2] == 1 else None)

                if has_zoom:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                ax.axis('off')

                self._update_histogram(params['stretch'], params['bp_ratio'],
                                       params['level_min'], params['level_max'])

                self.after(0, self._on_refresh_done)

            except Exception as e:
                self._log(f'Error: {e}')
                self._set_status(False)

        self._worker = threading.Thread(target=_do, daemon=True)
        self._worker.start()

    def _on_refresh_done(self):
        self._canvas.draw_idle()
        self._set_status(False)
        self._log('Ready')

    # ── HISTOGRAMME ───────────────────────────────────────────────────────────

    def _update_histogram(self, stretch, bp_ratio, level_min, level_max):
        ax = self._ax_h
        ax.clear()
        ax.set_facecolor('black')

        raw_stretch = AsinhStretch(a=stretch)(self.lab.calibrated)
        data = raw_stretch.ravel()
        useful = data[(data > 0.0001) & (data < 0.9999)]

        if len(useful) > 0:
            h_min, h_max = np.percentile(useful, [0.5, 99.5])
            ax.hist(useful, bins=100, range=(h_min, h_max), color='cyan', alpha=0.4)

            p25 = np.percentile(raw_stretch, 25)
            p50 = np.percentile(raw_stretch, 50)
            cutoff = p25 + ((p50 - p25) * (bp_ratio - 1.0) * 5)
            ax.axvline(cutoff, color='red', linestyle='-', linewidth=2, alpha=0.8)

            span = h_max - h_min
            ax.axvline(h_min + level_min * span, color='white',
                       linestyle='--', linewidth=1.5, alpha=0.9)
            ax.axvline(h_min + level_max * span, color='#ffeb3b',
                       linestyle='--', linewidth=1.5, alpha=0.9)

        ax.axis('off')
        self.after(0, self._canvas_h.draw_idle)

    # ── EXPORT ────────────────────────────────────────────────────────────────

    def _export(self):
        if self.lab.calibrated is None:
            self._log('No image loaded.')
            return

        save_dir = filedialog.askdirectory(
            title='Export to directory', initialdir=self._current_dir)
        if not save_dir:
            return

        self._set_status(True)
        self._log('Exporting…')

        settings = {s: self.sliders[s].value for s in self.sliders}
        settings['Level Min']        = self._sl_level_min.value
        settings['Level Max']        = self._sl_level_max.value
        settings['Gradient removal'] = self._var_grad.get()
        settings['Show Masks']       = self._var_masks.get()

        def _do():
            import imageio.v3 as imageio
            try:
                rgb = (settings['Red'], settings['Green'], settings['Blue'])
                img = self.lab.apply_cosmetics(
                    stretch      = settings['Stretch'],
                    bp_ratio     = settings['BlackPt'],
                    clarity      = settings['Clarity'],
                    denoise      = settings['Denoise'],
                    saturation   = settings['Saturation'],
                    galaxy_sigma = settings['Galaxy σ'],
                    stars_sigma  = settings['Stars σ'],
                    rgb          = rgb,
                    do_grad      = settings['Gradient removal'],
                )
                lmin, lmax = settings['Level Min'], settings['Level Max']
                if lmin > 0.0 or lmax < 1.0:
                    img = np.clip((img - lmin) / max(lmax - lmin, 1e-6), 0, 1)

                base = os.path.join(save_dir, f"photo_{int(time.time())}")

                # TIFF
                img16 = (np.squeeze(img) * 65535).astype(np.uint16)
                imageio.imwrite(base + '.tiff', img16, compression=None)

                # META
                lines = [
                    '# Photo Lab — export settings',
                    f'# {time.strftime("%Y-%m-%d %H:%M:%S")}',
                    f'# File : {base}.tiff', '',
                ]
                for key, val in settings.items():
                    lines.append(f'{key} = {val:.4f}' if isinstance(val, float)
                                 else f'{key} = {val}')
                with open(base + '.meta', 'w') as f:
                    f.write('\n'.join(lines) + '\n')

                self._log(f'Export done: {os.path.basename(base)}.tiff + .meta')
            except Exception as e:
                self._log(f'Export error: {e}')
            finally:
                self._set_status(False)

        self._worker = threading.Thread(target=_do, daemon=True)
        self._worker.start()


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = PhotoDashboardTk()
    app.mainloop()
