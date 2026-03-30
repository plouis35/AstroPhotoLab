"""
Photo_Lab.py
Module de traitement d'images astronomiques - module de traitement

v3 :
  1. MTF (Midtone Transfer Function) remplace AsinhStretch.
     Préserve mieux les hautes lumières (noyaux galactiques, étoiles brillantes).
     Paramètre m ∈ ]0,1[ : petit = stretch fort, grand = stretch doux.
     Défaut 0.15 ≈ stretch modéré, équivalent à l'ancien Stretch 0.01.

  2. Star Reduction par érosion morphologique.
     Réduit les halos post-stretch sans toucher à la nébuleuse.
     Utilise le masque étoiles existant comme guide.
     Paramètre star_reduce ∈ [0, 1], 0 = désactivé.

  3. ABE amélioré (Adaptive Background Extraction).
     Grille paramétrable (grid_size) au lieu d'une grille fixe 8x8.
     Mesure de fond par sigma-clipping dans chaque tuile au lieu du seul percentile 10.
     Plus robuste aux nébuleuses qui débordent dans une tuile.

v2 :
  - apply_cosmetics() ne recalcule plus les masques automatiquement.
  - load_files() invalide self.masks = {} après un nouveau stack.
  - Normalisation FITS via BITPIX.
  - Imports nettoyés.
  - parallélisation du chargement et de l'alignement (max_thread)

v1 :
  - conversion du notebook vers Qt6
"""

import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from astropy.io import fits
from scipy import ndimage
import imageio.v3 as imageio
from skimage.registration import phase_cross_correlation


# =============================================================================
class PhotoLab:
# =============================================================================

    def __init__(self, logger_callback=None):
        self.calibrated = None
        self.masks      = {}
        self.log        = logger_callback if logger_callback else print

    # =========================================================================
    # STRETCH — MTF (Midtone Transfer Function)
    # =========================================================================

    @staticmethod
    def _mtf(img, m):
        """
        Midtone Transfer Function — courbe sigmoïde standard PixInsight/Siril.

        f(x) = (m-1)·x / ((2m-1)·x - m)

        m ∈ ]0, 1[ :
          - m petit (0.05–0.15) → stretch fort, idéal pour nébuleuses faibles
          - m grand (0.4–0.5)   → stretch très doux, conserve les détails HH
        La fonction est définie en x=0 → 0 et x=1 → 1 (pas de clipping).
        Contrairement à AsinhStretch, les hautes lumières ne sont pas écrasées.
        """
        m  = float(np.clip(m, 1e-6, 1 - 1e-6))
        x  = np.clip(img, 0.0, 1.0)
        # Évite la division par zéro au point fixe x = m/(2m-1)
        denom = (2.0 * m - 1.0) * x - m
        # là où denom ≈ 0, la MTF tend vers l'infini → on clamp à 1
        safe  = np.where(np.abs(denom) < 1e-9, 1.0,
                         (m - 1.0) * x / denom)
        return np.clip(safe, 0.0, 1.0)

    # =========================================================================
    # ALIGNEMENT
    # =========================================================================

    def get_shift(self, ref_img, target_img, upsample_factor=10, star_percentile=99.0):
        """
        Calcul du décalage (y, x) par phase_cross_correlation (scikit-image).

        - upsample_factor  : précision sub-pixel (10 = 0.1 px).
        - star_percentile  : seuillage pour isoler les étoiles.
        """
        img_A = np.nan_to_num(ref_img.astype(float))
        img_B = np.nan_to_num(target_img.astype(float))

        thresh_A = np.percentile(img_A, star_percentile)
        thresh_B = np.percentile(img_B, star_percentile)
        clean_A  = np.clip(img_A - thresh_A, 0, None)
        clean_B  = np.clip(img_B - thresh_B, 0, None)

        shift, _error, _ = phase_cross_correlation(
            clean_A, clean_B,
            upsample_factor=upsample_factor,
            normalization=None
        )
        return shift   # array [dy, dx]

    # =========================================================================
    # CHARGEMENT
    # =========================================================================

    def _debayer_manual(self, img, pattern='RGGB'):
        """Dématriçage manuel optimisé (évite OpenCV). Pattern RGGB standard NINA/ZWO."""
        h, w = img.shape
        rgb  = np.zeros((h, w, 3), dtype=float)

        if pattern == 'RGGB':
            rgb[0::2, 0::2, 0] = img[0::2, 0::2]
            rgb[1::2, 1::2, 2] = img[1::2, 1::2]
            rgb[0::2, 1::2, 1] = img[0::2, 1::2]
            rgb[1::2, 0::2, 1] = img[1::2, 0::2]

        k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) * 0.25
        r, b = rgb[:, :, 0], rgb[:, :, 2]
        rgb[:, :, 0] = r + ndimage.convolve(r, k) * 4 * (r == 0)
        rgb[:, :, 2] = b + ndimage.convolve(b, k) * 4 * (b == 0)
        g            = rgb[:, :, 1]
        missing_g    = (rgb[:, :, 0] > 0) | (rgb[:, :, 2] > 0)
        rgb[:, :, 1] = g + ndimage.convolve(g, k) * 4 * missing_g

        return rgb

    def _smart_load(self, path, is_master=False):
        """Charge TIFF, PNG ou FITS → image normalisée [0, 1]."""
        ext = os.path.splitext(path)[1].lower()

        if ext in ('.fits', '.fit'):
            with fits.open(path) as hdul:
                hdr    = hdul[0].header
                raw    = hdul[0].data.astype(float)
                bitpix  = abs(hdr.get('BITPIX', 16))
                max_val = (2 ** bitpix) - 1 if bitpix in (8, 12, 14, 16, 32) else 65535
                if np.max(raw) > 1.0:
                    raw /= float(max_val)
                raw = np.flipud(raw)
                if raw.ndim == 2 and not is_master and 'BAYERPAT' in hdr:
                    pattern = hdr['BAYERPAT'].replace("'", "").strip()
                    if len(pattern) != 4:
                        pattern = 'RGGB'
                    return self._debayer_manual(raw, pattern)
                return raw
        else:
            raw   = imageio.imread(path).astype(float)
            limit = 65535.0 if np.max(raw) > 255 else 255.0
            return raw / limit

    def load_files(self, file_paths, max_workers=1):
        """Charge, calibre, aligne et stacke une liste de fichiers."""
        if not file_paths:
            return

        self.dark, self.flat = 0, 1
        base_dir = os.path.dirname(file_paths[0])
        for f in glob.glob(os.path.join(base_dir, "master*")):
            try:
                norm = self._smart_load(f, is_master=True)
                name = os.path.basename(f).lower()
                if "dark" in name: self.dark = norm
                if "flat" in name: self.flat = norm / (np.mean(norm) + 1e-10)
            except Exception:
                pass

        file_paths = sorted(p for p in file_paths
                            if "master" not in os.path.basename(p).lower())
        if not file_paths:
            return

        self.log(f"Loading {len(file_paths)} image(s) in parallel...")

        def _load_and_calib(args):
            i, p  = args
            raw   = self._smart_load(p, is_master=False)
            calib = (raw - self.dark) / (self.flat + 1e-6)
            return i, p, calib

        calibrated_imgs = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_load_and_calib, (i, p)): i
                       for i, p in enumerate(file_paths)}
            for fut in as_completed(futures):
                try:
                    i, p, calib = fut.result()
                    calibrated_imgs[i] = (p, calib)
                    self.log(f"Loaded {i+1}/{len(file_paths)}: {os.path.basename(p)}")
                except Exception as e:
                    self.log(f"Load error: {e}")

        if not calibrated_imgs:
            return

        _, ref_full  = calibrated_imgs[0]
        target_shape = ref_full.shape
        ref_chan      = ref_full[:, :, 1] if ref_full.ndim == 3 else ref_full

        self.log("Computing shifts in parallel...")

        def _compute_shift(args):
            i, calib = args
            if calib.shape != target_shape:
                if calib.shape[0] == target_shape[1]:
                    calib = np.rot90(calib)
                else:
                    return i, None, None
            img_B = calib[:, :, 1] if calib.ndim == 3 else calib
            shift = self.get_shift(ref_chan, img_B)
            return i, shift, calib

        shifts  = {}
        indices = sorted(calibrated_imgs.keys())[1:]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_compute_shift, (i, calibrated_imgs[i][1])): i
                       for i in indices}
            for fut in as_completed(futures):
                try:
                    i, shift, calib = fut.result()
                    if shift is None:
                        self.log(f"Image {i}: skipped (shape mismatch)")
                        continue
                    if abs(shift[0]) > 200 or abs(shift[1]) > 200:
                        self.log(f"Image {i}: rejected (shift too large: "
                                 f"{shift[0]:.0f}, {shift[1]:.0f})")
                        continue
                    self.log(f"Image {i}: dy={shift[0]:.1f}, dx={shift[1]:.1f}")
                    shifts[i] = (shift, calib)
                except Exception as e:
                    self.log(f"Shift error {e}")

        self.log(f"Aligning and stacking {1 + len(shifts)} images...")

        def _apply_shift(calib, shift):
            if calib.ndim == 3:
                with ThreadPoolExecutor(max_workers=3) as ex:
                    futs     = [ex.submit(ndimage.shift, calib[:, :, c], shift, order=1)
                                for c in range(3)]
                    channels = [f.result() for f in futs]
                return np.stack(channels, axis=-1)
            return ndimage.shift(calib, shift, order=1)

        stacked   = ref_full.copy().astype(np.float64)
        n_stacked = 1
        for i in sorted(shifts.keys()):
            shift, calib = shifts[i]
            aligned       = _apply_shift(calib, shift)
            stacked      += aligned
            n_stacked    += 1
            del aligned
            shifts[i] = None

        self.log(f"Stacking done ({n_stacked} images). Normalizing...")
        bp_pc = 1
        if stacked.ndim == 3:
            for c in range(3):
                stacked[:, :, c] -= np.percentile(stacked[:, :, c], bp_pc)
        else:
            stacked -= np.percentile(stacked, bp_pc)

        p_high          = np.percentile(stacked, 99.9)
        self.calibrated = np.clip(stacked / (p_high + 1e-10), 0, 1)
        self.masks      = {}

        #if self.calibrated.ndim == 2:
        #    self.calibrated = self.calibrated[:, :, None]
        if self.calibrated.ndim == 2:
            self.calibrated = np.stack([self.calibrated]*3, axis=-1)  # shape (h, w, 3)
        elif self.calibrated.shape[2] == 1:
            self.calibrated = np.concatenate([self.calibrated]*3, axis=-1)

        self.log(f"Ready — {n_stacked} image(s) stacked")

    # =========================================================================
    # ABE AMÉLIORÉ — Adaptive Background Extraction
    # =========================================================================

    def remove_gradient(self, img, grid_size=16):
        """
        Soustraction du gradient de fond par grille adaptative.

        Améliorations v3 par rapport à la grille 8×8 fixe :
          - grid_size paramétrable (défaut 16×16, soit 4× plus de points de mesure).
          - Mesure de fond par sigma-clipping dans chaque tuile : on rejette les pixels
            > mean + 2σ (étoiles, nébuleuse) avant de prendre le percentile 20.
            Beaucoup plus robuste quand une nébuleuse déborde dans une tuile.
          - Rééquilibrage final inchangé.
        """
        self.log("Removing gradient (ABE)...")
        try:
            h, w   = img.shape[:2]
            out    = img.copy()
            step_y = max(h // grid_size, 1)
            step_x = max(w // grid_size, 1)

            for c in range(img.shape[2]):
                chan         = img[:, :, c]
                bg_map_small = np.zeros((grid_size, grid_size))

                for gy in range(grid_size):
                    for gx in range(grid_size):
                        tile = chan[gy*step_y:(gy+1)*step_y,
                                    gx*step_x:(gx+1)*step_x].ravel()
                        if tile.size == 0:
                            continue
                        # Sigma-clipping : on rejette les pixels brillants (étoiles)
                        mean, std = tile.mean(), tile.std()
                        sky = tile[tile < mean + 2.0 * std]
                        if sky.size < 4:
                            sky = tile          # fallback si tuile trop "pleine"
                        bg_map_small[gy, gx] = np.percentile(sky, 20)

                bg_model     = ndimage.zoom(bg_map_small,
                                            (h / grid_size, w / grid_size),
                                            order=3)
                bg_model     = bg_model[:h, :w]
                out[:, :, c] = np.clip(chan - bg_model, 0, 1)

            return out - np.min(out)
        except Exception:
            return img

    # =========================================================================
    # MASQUES (V3 : Morphologie & Protection)
    # =========================================================================

    def _make_star_mask_v3_dilated(self, gray_img, sensitivity=3.0):
        """Détecte les étoiles (high-pass) et dilate pour couvrir les halos."""
        low_freq    = ndimage.gaussian_filter(gray_img, sigma=8)
        high_freq   = gray_img - low_freq
        binary_mask = high_freq > np.std(high_freq) * sensitivity
        return ndimage.binary_dilation(binary_mask, iterations=2).astype(float)

    def _make_galaxy_mask_v3_morpho(self, gray_img, galaxy_sigma):
        """Isole la nébuleuse en supprimant d'abord les étoiles."""
        if galaxy_sigma <= 0.1:
            return np.zeros_like(gray_img)
        starless        = ndimage.grey_opening(gray_img, size=(5, 5))
        smooth_starless = ndimage.gaussian_filter(starless, sigma=4)
        threshold       = np.mean(smooth_starless) + 0.5 * np.std(smooth_starless)
        return (smooth_starless > threshold).astype(float)

    def extract_masks(self, img, stars_sigma=15, galaxy_sigma=20):
        """
        Calcule et stocke les masques étoiles / galaxie / fond.
        Appelé par le Dashboard uniquement quand stars_sigma / galaxy_sigma changent.
        """
        gray     = np.mean(img, axis=2)
        raw_star = self._make_star_mask_v3_dilated(gray, sensitivity=3.0)
        s_mask   = ndimage.gaussian_filter(raw_star, sigma=stars_sigma)
        raw_gal  = self._make_galaxy_mask_v3_morpho(gray, galaxy_sigma)
        if np.max(raw_gal) > 0:
            star_shield = ndimage.binary_dilation(raw_star, iterations=4).astype(float)
            raw_gal     = np.clip(raw_gal - star_shield, 0, 1)
        g_mask = ndimage.gaussian_filter(raw_gal, sigma=galaxy_sigma)
        self.masks = {
            'stars':      np.clip(s_mask, 0, 1),
            'galaxy':     np.clip(g_mask, 0, 1),
            'background': np.clip(1.0 - g_mask - s_mask, 0, 1),
        }

    # =========================================================================
    # ÉTAPES COSMÉTIQUES
    # =========================================================================

    def _step_star_reduction(self, img, amount):
        """
        Star Reduction v1 — érosion morphologique guidée par le masque étoiles.

        Mécanisme :
          1. On érode l'image entière (rétrécit tous les objets brillants).
          2. On mélange l'image érodée avec l'originale, pondéré par le masque étoiles.
             → Seules les zones identifiées comme étoiles sont affectées.
          3. Un léger lissage gaussien sur le résultat adoucit les bords de l'érosion.

        amount ∈ [0, 1] :
          - 0   → aucun effet
          - 0.3 → réduction légère, recommandée pour images bien résolues
          - 0.7 → réduction forte, utile sur champs très denses
        """
        if amount <= 0 or 'stars' not in self.masks:
            return img

        # Taille du structurant : 1 pixel à amount=0, 5 pixels à amount=1
        struct_size = max(1, int(round(amount * 5)))
        struct      = ndimage.generate_binary_structure(2, 1)
        if struct_size > 1:
            struct = ndimage.iterate_structure(struct, struct_size)

        eroded = np.zeros_like(img)
        for c in range(img.shape[2]):
            eroded[:, :, c] = ndimage.grey_erosion(img[:, :, c],
                                                    footprint=struct)

        # Mélange guidé par le masque étoiles
        star_mask = self.masks['stars'][..., None]
        result    = img * (1.0 - star_mask * amount) + eroded * (star_mask * amount)

        # Adoucissement léger sur les zones étoiles pour éviter les artefacts de bord
        blur   = ndimage.gaussian_filter(result, sigma=(0.8, 0.8, 0))
        result = result * (1.0 - star_mask * 0.3) + blur * (star_mask * 0.3)

        return np.clip(result, 0, 1)

    def _step_denoise_harmonized(self, img, amount, galaxy_sigma):
        """Denoise RGB : lissage fort sur le fond, léger sur l'objet."""
        if amount <= 0:
            return img
        img_strong = ndimage.gaussian_filter(img, sigma=(amount, amount, 0))
        if galaxy_sigma > 0.0 and 'background' in self.masks:
            img_light = ndimage.gaussian_filter(img, sigma=(amount * 0.4, amount * 0.4, 0))
            m_back    = self.masks['background'][..., None]
            return img_strong * m_back + img_light * (1.0 - m_back)
        return img_strong

    def _step_clarity_multiscale(self, img, strength, galaxy_sigma):
        """Clarity Volume : Mélange détails fins et moyens."""
        if strength <= 0:
            return img
        fine      = img - ndimage.gaussian_filter(img, sigma=(2, 2, 0))
        medium    = img - ndimage.gaussian_filter(img, sigma=(8, 8, 0))
        structure = fine * 0.5 + medium * 1.5
        if galaxy_sigma > 0.1 and 'galaxy' in self.masks:
            return img + structure * strength * self.masks['galaxy'][..., None]
        return img + structure * strength

    def _step_dynamic_compression(self, img, bp_strength, galaxy_sigma):
        """Compression Dynamique (Black Point) : ancrage + gamma sélectif."""
        if self.calibrated is None:
            return img
        img = np.clip(img, 0, 1)
        if galaxy_sigma > 0.1 and 'background' in self.masks:
            luma    = np.mean(img, axis=2)
            bg_vals = luma[self.masks['background'] > 0.5]
            if len(bg_vals) > 0:
                pedestal = np.percentile(bg_vals, 5)
                img      = np.clip(img - pedestal, 0, 1)
        gamma_strength = 1.0 + (bp_strength - 1.0) * 2.0
        if galaxy_sigma > 0.1 and 'background' in self.masks:
            gamma_map = 1.0 + (gamma_strength - 1.0) * self.masks['background']
            return np.power(img, gamma_map[..., None])
        return np.power(img, gamma_strength)

    def _step_saturation_smart(self, img, sat_amount, galaxy_sigma):
        """Saturation HSV sélective (protège le fond)."""
        if sat_amount == 1.0:
            return img
        hsv = rgb_to_hsv(np.clip(img, 0, 1))
        if galaxy_sigma > 0.0 and 'galaxy' in self.masks:
            target_sat    = 0.5 + (sat_amount - 0.5) * self.masks['galaxy']
            hsv[:, :, 1] *= target_sat
        else:
            hsv[:, :, 1] *= sat_amount
        return hsv_to_rgb(np.clip(hsv, 0, 1))

    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================

    def apply_cosmetics(self, mtf, bp_ratio, clarity, denoise, saturation,
                        galaxy_sigma, stars_sigma, star_reduce=0.0,
                        rgb=(1, 1, 1), do_grad=True, grad_grid=16):
        """
        Applique la chaîne cosmétique complète sur self.calibrated.

        Paramètres v3 :
          mtf         : midtone ∈ ]0,1[, remplace 'stretch'. Défaut 0.15.
          star_reduce : réduction halos étoiles ∈ [0,1]. Défaut 0 = off.
          grad_grid   : taille de la grille ABE. Défaut 16.

        NOTE : les masques ne sont PAS recalculés ici. C'est le Dashboard
        qui appelle extract_masks() via le flag update_masks.
        """
        if self.calibrated is None:
            return None

        img = self.calibrated.copy()

        # 1. ABE — suppression du gradient de fond
        if do_grad:
            img = self.remove_gradient(img, grid_size=grad_grid)

        # 2. Balance RGB
        self.log("Balancing colors...")
        for i in range(3):
            img[:, :, i] *= rgb[i]

        # 3a. Denoise
        if denoise > 0:
            self.log("Denoise...")
            img = self._step_denoise_harmonized(img, denoise, galaxy_sigma)

        # 3b. Clarity
        if clarity > 0:
            self.log("Clarity...")
            img = self._step_clarity_multiscale(img, clarity, galaxy_sigma)

        # 3c. Black point
        if bp_ratio != 1.0:
            self.log("Black point...")
            img = self._step_dynamic_compression(img, bp_ratio, galaxy_sigma)

        # 3d. Saturation
        if saturation > 0:
            self.log("Color saturation...")
            img = self._step_saturation_smart(img, saturation, galaxy_sigma)

        # 4. Stretch MTF
        self.log("Stretching (MTF)...")
        img = self._mtf(img, mtf)

        # 5. Star reduction (après stretch — c'est là que les halos apparaissent)
        if star_reduce > 0 and 'stars' in self.masks:
            self.log("Star reduction...")
            img = self._step_star_reduction(img, star_reduce)

        return np.clip(img, 0, 1)
