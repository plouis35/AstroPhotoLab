"""
Photo_Dashboard.py
Application de retouche photo astronomique.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QGroupBox, QFileDialog,
    QSizePolicy, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPalette, QFont

from Photo_Lab import PhotoLab


class Worker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn     = fn
        self._args   = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class AstroToolbar(NavigationToolbar2QT):
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] in ('Home', 'Back', 'Forward', None, 'Pan', 'Zoom')]

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.setStyleSheet("""
            QToolBar { background: #2d2d2d; border: none; spacing: 2px; padding: 2px; }
            QToolButton { background: #3d3d3d; border: 1px solid #555; border-radius: 4px; padding: 3px; color: #eee; }
            QToolButton:hover   { background: #555; }
            QToolButton:checked { background: #1565c0; border-color: #1976d2; }
        """)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(9, 7), facecolor='black', toolbar=False):
        self.fig = Figure(figsize=figsize, facecolor=facecolor)
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax  = self.fig.add_subplot(111)
        self.ax.axis('off')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label, vmin, vmax, vstep, vdefault, handle_color=None):
        super().__init__()
        self._min   = vmin
        self._max   = vmax
        self._step  = vstep
        self._steps = round((vmax - vmin) / vstep)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setFixedWidth(80)
        lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self._steps)
        self.slider.setValue(self._val_to_int(vdefault))
        self.slider.setTickInterval(1)
        if handle_color:
            self.slider.setStyleSheet(f"""
                QSlider::handle:horizontal {{
                    background: {handle_color}; border: 1px solid #888;
                    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
                }}
                QSlider::groove:horizontal {{ background: #444; height: 4px; border-radius: 2px; }}
                QSlider::sub-page:horizontal {{ background: {handle_color}; height: 4px; border-radius: 2px; }}
            """)
        layout.addWidget(self.slider)

        self.val_lbl = QLabel(f"{vdefault:.3g}")
        self.val_lbl.setFixedWidth(48)
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_lbl.setStyleSheet("color: #fff; font-size: 11px; font-family: monospace;")
        layout.addWidget(self.val_lbl)

        self.slider.valueChanged.connect(self._on_drag)
        self.slider.sliderReleased.connect(self._on_release)

    def _val_to_int(self, v):
        return round((v - self._min) / self._step)

    def _int_to_val(self, i):
        return self._min + i * self._step

    def _on_drag(self, i):
        self.val_lbl.setText(f"{self._int_to_val(i):.3g}")

    def _on_release(self):
        self.valueChanged.emit(self.value)

    @property
    def value(self):
        return self._int_to_val(self.slider.value())

    def setValue(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(self._val_to_int(v))
        self.val_lbl.setText(f"{v:.3g}")
        self.slider.blockSignals(False)


class PhotoDashboardQt(QMainWindow):

    _log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Lab — Astronomical Image Processing")
        self.resize(1400, 860)
        self._apply_dark_theme()

        self._log_signal.connect(self._update_log_label)

        self.lab     = PhotoLab()
        self.lab.log = self._emit_log

        self._current_dir     = os.getcwd()
        self._workers         = []
        self._pending_refresh = None

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._do_pending_refresh)

        self._build_ui()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1a1a2e; color: #e0e0e0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 8px; font-size: 11px; font-weight: bold; color: #aaa; padding: 6px 4px 4px 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #00bcd4; }
            QPushButton { background-color: #2d2d44; border: 1px solid #555; border-radius: 4px; padding: 5px 12px; color: #e0e0e0; }
            QPushButton:hover   { background-color: #3d3d5c; }
            QPushButton:pressed { background-color: #1a1a30; }
            QPushButton#btn_load      { background-color: #0d47a1; border-color: #1565c0; }
            QPushButton#btn_load:hover { background-color: #1565c0; }
            QPushButton#btn_exp       { background-color: #1b5e20; border-color: #2e7d32; }
            QPushButton#btn_exp:hover  { background-color: #2e7d32; }
            QListWidget { background-color: #12122a; border: 1px solid #444; color: #ccc; font-size: 11px; }
            QListWidget::item:selected { background-color: #0d47a1; color: white; }
            QSlider::groove:horizontal { background: #444; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #888; border: 1px solid #aaa; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #1565c0; height: 4px; border-radius: 2px; }
            QCheckBox { color: #e0e0e0; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 2px solid #888; border-radius: 3px; background-color: #12122a; }
            QCheckBox::indicator:hover   { border-color: #00bcd4; }
            QCheckBox::indicator:checked { background-color: #1565c0; border-color: #42a5f5; image: url(none); }
            QCheckBox::indicator:checked:hover { background-color: #1976d2; }
            QLabel#dir_label { color: #888; font-size: 10px; background: #12122a; border: 1px solid #333; padding: 3px 6px; border-radius: 3px; }
            QScrollArea { border: none; }
        """)

    def _build_ui(self):
        central     = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(300)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)
        left_scroll.setWidget(left_widget)

        # -- FILES --
        grp_files        = QGroupBox("FILES")
        grp_files_layout = QVBoxLayout(grp_files)
        self.files_label = QLabel("No file loaded")
        self.files_label.setObjectName("dir_label")
        self.files_label.setWordWrap(True)
        grp_files_layout.addWidget(self.files_label)
        btn_row       = QHBoxLayout()
        self.btn_load = QPushButton("⬇  Load…")
        self.btn_load.setObjectName("btn_load")
        self.btn_load.clicked.connect(self._on_load_click)
        self.btn_exp  = QPushButton("💾  Export")
        self.btn_exp.setObjectName("btn_exp")
        self.btn_exp.clicked.connect(self._export)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_exp)
        grp_files_layout.addLayout(btn_row)
        left_layout.addWidget(grp_files)

        # -- DISPLAY --
        grp_view        = QGroupBox("DISPLAY")
        grp_view_layout = QVBoxLayout(grp_view)
        self.chk_grad   = QCheckBox("Gradient removal (ABE)")
        self.chk_grad.toggled.connect(
            lambda v: self._refresh(False, f"Gradient removal: {'ON' if v else 'OFF'}")
        )
        self.chk_masks  = QCheckBox("Show masks")
        self.chk_masks.toggled.connect(
            lambda v: self._refresh(False, f"Show masks: {'ON' if v else 'OFF'}")
        )
        chk_row = QHBoxLayout()
        chk_row.addWidget(self.chk_grad)
        chk_row.addWidget(self.chk_masks)
        grp_view_layout.addLayout(chk_row)

        # Slider ABE Grid (8–32, pas 4, défaut 16)
        self.sl_abe_grid = LabeledSlider('ABE grid', 8, 32, 4, 16)
        self.sl_abe_grid.valueChanged.connect(lambda v: self._refresh(False, "ABE grid"))
        grp_view_layout.addWidget(self.sl_abe_grid)
        left_layout.addWidget(grp_view)

        # -- HISTOGRAM --
        grp_hist        = QGroupBox("HISTOGRAM  (red = black point)")
        grp_hist_layout = QVBoxLayout(grp_hist)
        grp_hist_layout.setContentsMargins(2, 2, 2, 2)
        self.hist_canvas = MplCanvas(figsize=(2.8, 1.6), facecolor='black')
        self.hist_canvas.setFixedHeight(110)
        grp_hist_layout.addWidget(self.hist_canvas)
        self.sl_level_min = LabeledSlider('Min', 0.0, 0.5, 0.001, 0.0, handle_color='#ffffff')
        self.sl_level_max = LabeledSlider('Max', 0.5, 1.0, 0.001, 1.0, handle_color='#ffeb3b')
        self.sl_level_min.valueChanged.connect(lambda v: self._refresh(False, "Level Min"))
        self.sl_level_max.valueChanged.connect(lambda v: self._refresh(False, "Level Max"))
        grp_hist_layout.addWidget(self.sl_level_min)
        grp_hist_layout.addWidget(self.sl_level_max)
        left_layout.addWidget(grp_hist)

        # -- SLIDERS --
        self.sliders = {}
        slider_defs  = [
            ("COLOR BALANCE", [
                ('Red',   0.5, 2.0, 0.05, 1.0, 'red'),
                ('Green', 0.5, 2.0, 0.05, 1.0, 'lime'),
                ('Blue',  0.5, 2.0, 0.05, 1.0, '#4fc3f7'),
            ]),
            ("MASKS", [
                ('Stars σ',  0.0, 20.0, 0.5, 0.0, None),
                ('Galaxy σ', 0.0, 80.0, 0.5, 0.0, None),
            ]),
            ("COSMETICS", [
                # MTF : valeur faible = stretch fort (sens "naturel" pour l'astro)
                # Plage 0.02–0.5, défaut 0.15 ≈ stretch modéré
                ('MTF',        0.02, 0.50, 0.01, 0.15, None),
                ('BlackPt',    0.0,  2.0,  0.01, 1.0,  None),
                ('Clarity',    0.0,  2.0,  0.1,  0.0,  None),
                ('Denoise',    0.0,  2.0,  0.1,  0.0,  None),
                ('Saturation', 0.0,  3.0,  0.1,  0.0,  None),
                # StarRed : 0 = désactivé, agir après stretch uniquement
                ('StarRed',    0.0,  1.0,  0.05, 0.0,  None),
            ]),
        ]
        for grp_name, items in slider_defs:
            grp        = QGroupBox(grp_name)
            grp_layout = QVBoxLayout(grp)
            grp_layout.setSpacing(2)
            for name, vmin, vmax, vstep, vdef, hcolor in items:
                s       = LabeledSlider(name, vmin, vmax, vstep, vdef, handle_color=hcolor)
                is_mask = name in ('Stars σ', 'Galaxy σ')
                s.valueChanged.connect(
                    lambda v, n=name, m=is_mask: self._refresh(m, f"Setting: {n}")
                )
                grp_layout.addWidget(s)
                self.sliders[name] = s
            left_layout.addWidget(grp)

        left_layout.addStretch()

        # -- PANNEAU DROIT --
        right_layout    = QVBoxLayout()
        right_layout.setSpacing(4)
        status_bar      = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(18)
        self._set_status(False)
        self.log_label  = QLabel("Ready")
        self.log_label.setStyleSheet("color: #aaa; font-size: 11px;")
        status_bar.addWidget(self.status_dot)
        status_bar.addWidget(self.log_label)
        status_bar.addStretch()
        right_layout.addLayout(status_bar)
        self.main_canvas = MplCanvas(figsize=(9, 7), facecolor='black')
        toolbar          = AstroToolbar(self.main_canvas, self)
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.main_canvas)

        main_layout.addWidget(left_scroll)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget, stretch=1)

    # ── LOGGING THREAD-SAFE ───────────────────────────────────────────────────

    def _emit_log(self, msg: str):
        self._log_signal.emit(str(msg))

    def _update_log_label(self, msg: str):
        self.log_label.setText(msg)

    def _log(self, msg: str):
        self._emit_log(msg)

    def _set_status(self, busy: bool):
        color = "#e74c3c" if busy else "#2ecc71"
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 18px;")

    # ── GESTION DES WORKERS ───────────────────────────────────────────────────

    def _register_worker(self, worker):
        self._workers.append(worker)
        worker.finished.connect(lambda _: self._cleanup_workers())
        worker.error.connect(lambda _: self._cleanup_workers())
        return worker

    def _cleanup_workers(self):
        self._workers = [w for w in self._workers if w.isRunning()]

    def _active_worker(self):
        return self._workers[-1] if self._workers and self._workers[-1].isRunning() else None

    # ── CHARGEMENT ────────────────────────────────────────────────────────────

    def _on_load_click(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select image files", self._current_dir,
            "Images (*.png *.tif *.tiff *.fit *.fits)"
        )
        if not paths:
            return
        self._current_dir = os.path.dirname(paths[0])

        names = [os.path.basename(p) for p in paths]
        def _trunc(s, n=22):
            return s[:n] + '…' if len(s) > n else s
        if len(names) <= 3:
            summary = ', '.join(_trunc(n) for n in names)
        else:
            summary = ', '.join(_trunc(n) for n in names[:3]) + f'  (+{len(names)-3} more)'
        self.files_label.setText(f"{len(paths)} file(s) :\n{summary}")

        self._set_status(True)
        self._log("Loading files…")
        self.btn_load.setEnabled(False)

        w = Worker(self.lab.load_files, paths)
        w.finished.connect(self._on_load_done)
        w.error.connect(lambda e: (self._log(f"Error: {e}"), self._set_status(False)))
        self._register_worker(w)
        w.start()

    def _on_load_done(self, _):
        self.btn_load.setEnabled(True)
        self._refresh(True, "Refreshing image…")

    # ── REFRESH / RENDU ───────────────────────────────────────────────────────

    def _refresh(self, update_masks=False, log_msg=None):
        if self.lab.calibrated is None:
            return
        prev = self._pending_refresh
        self._pending_refresh = (
            update_masks or (prev[0] if prev else False),
            log_msg      or (prev[1] if prev else None),
        )
        self._debounce_timer.start(300)

    def _do_pending_refresh(self):
        if self._pending_refresh is None:
            return
        if self._active_worker() is not None:
            self._debounce_timer.start(200)
            return

        update_masks, log_msg = self._pending_refresh
        self._pending_refresh = None

        if log_msg:
            self._log(log_msg)
        self._set_status(True)

        params = dict(
            update_masks = update_masks,
            show_masks   = self.chk_masks.isChecked(),
            mtf          = self.sliders['MTF'].value,
            bp_ratio     = self.sliders['BlackPt'].value,
            clarity      = self.sliders['Clarity'].value,
            denoise      = self.sliders['Denoise'].value,
            saturation   = self.sliders['Saturation'].value,
            star_reduce  = self.sliders['StarRed'].value,
            galaxy_sigma = self.sliders['Galaxy σ'].value,
            stars_sigma  = self.sliders['Stars σ'].value,
            rgb          = (self.sliders['Red'].value,
                            self.sliders['Green'].value,
                            self.sliders['Blue'].value),
            do_grad      = self.chk_grad.isChecked(),
            grad_grid    = int(self.sl_abe_grid.value),
            level_min    = self.sl_level_min.value,
            level_max    = self.sl_level_max.value,
        )

        def _do_refresh():
            if params['update_masks'] or not self.lab.masks:
                self.lab.extract_masks(
                    self.lab.calibrated,
                    params['stars_sigma'],
                    params['galaxy_sigma'],
                )

            ax = self.main_canvas.ax
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            has_zoom   = xlim not in [(0.0, 1.0), (0, 1)]
            ax.clear()

            if params['show_masks'] and self.lab.masks:
                m  = np.zeros((*self.lab.calibrated.shape[:2], 3))
                m += self.lab.masks['stars'][:, :, None]      * [1, 0, 0]
                if params['galaxy_sigma'] > 0:
                    m += self.lab.masks['galaxy'][:, :, None]     * [0, 1, 0]
                    m += self.lab.masks['background'][:, :, None] * [0, 0, 0.2]
                ax.imshow(m)
            else:
                img = self.lab.apply_cosmetics(
                    mtf          = params['mtf'],
                    bp_ratio     = params['bp_ratio'],
                    clarity      = params['clarity'],
                    denoise      = params['denoise'],
                    saturation   = params['saturation'],
                    galaxy_sigma = params['galaxy_sigma'],
                    stars_sigma  = params['stars_sigma'],
                    star_reduce  = params['star_reduce'],
                    rgb          = params['rgb'],
                    do_grad      = params['do_grad'],
                    grad_grid    = params['grad_grid'],
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
            self._update_histogram(params['mtf'], params['bp_ratio'],
                                   params['level_min'], params['level_max'])
            return None

        w = Worker(_do_refresh)
        w.finished.connect(self._on_refresh_done)
        w.error.connect(lambda e: (self._log(f"Error: {e}"), self._set_status(False)))
        self._register_worker(w)
        w.start()

    def _on_refresh_done(self, _):
        self.main_canvas.fig.canvas.draw_idle()
        self._set_status(False)
        self._log("Ready")

    def _update_histogram(self, mtf=None, bp_ratio=None, level_min=None, level_max=None):
        if mtf      is None: mtf       = self.sliders['MTF'].value
        if bp_ratio is None: bp_ratio  = self.sliders['BlackPt'].value
        if level_min is None: level_min = self.sl_level_min.value
        if level_max is None: level_max = self.sl_level_max.value

        ax_h = self.hist_canvas.ax
        ax_h.clear()
        ax_h.set_facecolor('black')

        # Prévisualisation MTF sur l'image calibrée
        stretched = PhotoLab._mtf(self.lab.calibrated, mtf)
        data      = stretched.ravel()
        useful    = data[(data > 0.0001) & (data < 0.9999)]

        if len(useful) > 0:
            h_min, h_max = np.percentile(useful, [0.5, 99.5])
            ax_h.hist(useful, bins=100, range=(h_min, h_max), color='cyan', alpha=0.4)

            p25      = np.percentile(stretched, 25)
            p50      = np.percentile(stretched, 50)
            std_fond = p50 - p25
            cutoff   = p25 + std_fond * (bp_ratio - 1.0) * 5
            ax_h.axvline(cutoff, color='red', linestyle='-', linewidth=2, alpha=0.8)

            span      = h_max - h_min
            vline_min = h_min + level_min * span
            vline_max = h_min + level_max * span
            ax_h.axvline(vline_min, color='white',   linestyle='--', linewidth=1.5, alpha=0.9)
            ax_h.axvline(vline_max, color='#ffeb3b', linestyle='--', linewidth=1.5, alpha=0.9)

        ax_h.axis('off')
        self.hist_canvas.fig.canvas.draw_idle()

    # ── EXPORT ────────────────────────────────────────────────────────────────

    def _export(self):
        if self.lab.calibrated is None:
            self._log("No image loaded.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Export to directory", self._current_dir)
        if not save_dir:
            return

        self._set_status(True)
        self._log("Exporting…")

        settings = {s: self.sliders[s].value for s in self.sliders}
        settings['Level Min']        = self.sl_level_min.value
        settings['Level Max']        = self.sl_level_max.value
        settings['ABE Grid']         = int(self.sl_abe_grid.value)
        settings['Gradient removal'] = self.chk_grad.isChecked()
        settings['Show Masks']       = self.chk_masks.isChecked()

        def _do_export():
            import imageio.v3 as imageio
            rgb = (settings['Red'], settings['Green'], settings['Blue'])
            img = self.lab.apply_cosmetics(
                mtf          = settings['MTF'],
                bp_ratio     = settings['BlackPt'],
                clarity      = settings['Clarity'],
                denoise      = settings['Denoise'],
                saturation   = settings['Saturation'],
                galaxy_sigma = settings['Galaxy σ'],
                stars_sigma  = settings['Stars σ'],
                star_reduce  = settings['StarRed'],
                rgb          = rgb,
                do_grad      = settings['Gradient removal'],
                grad_grid    = settings['ABE Grid'],
            )
            lmin, lmax = settings['Level Min'], settings['Level Max']
            if lmin > 0.0 or lmax < 1.0:
                img = np.clip((img - lmin) / max(lmax - lmin, 1e-6), 0, 1)

            base  = os.path.join(save_dir, f"photo_{int(time.time())}")
            img16 = (np.squeeze(img) * 65535).astype(np.uint16)
            imageio.imwrite(base + '.tiff', img16, compression=None)

            lines = [
                "# Photo Lab — export settings",
                f"# {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"# File : {base}.tiff",
                "",
            ]
            for key, val in settings.items():
                lines.append(f"{key} = {val:.4f}" if isinstance(val, float) else f"{key} = {val}")
            with open(base + '.meta', 'w') as f:
                f.write('\n'.join(lines) + '\n')

            return base

        w = Worker(_do_export)
        w.finished.connect(lambda b: (
            self._log(f"Export done: {b}.tiff  +  {os.path.basename(b)}.meta"),
            self._set_status(False),
        ))
        w.error.connect(lambda e: (self._log(f"Export error: {e}"), self._set_status(False)))
        self._register_worker(w)
        w.start()


if __name__ == "__main__":
    plt.style.use('dark_background')
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PhotoDashboardQt()
    win.show()
    sys.exit(app.exec_())
