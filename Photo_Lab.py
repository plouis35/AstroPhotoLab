import os, glob, time, functools
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb 
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch
from photutils.segmentation import detect_threshold, detect_sources
from scipy import ndimage
from PIL import Image
import imageio.v3 as imageio
from skimage.registration import phase_cross_correlation

class PhotoLab:
    def __init__(self, logger_callback=None):
        self.calibrated = None
        self.masks = {}
        self.log = logger_callback if logger_callback else print


    def get_shift(self, ref_img, target_img, upsample_factor=10, star_percentile=99.0):
        """
        Calcul du décalage (y, x) par phase_cross_correlation (scikit-image).
        
        - upsample_factor : précision sub-pixel (10 = 0.1px). Augmenter ralentit.
        - star_percentile : seuillage pour isoler les étoiles avant corrélation.
          Réduire (ex: 95) si les images ont peu d'étoiles brillantes.
        """
        img_A = np.nan_to_num(ref_img.astype(float))
        img_B = np.nan_to_num(target_img.astype(float))

        # Isolation des étoiles : on ne garde que les pixels au-dessus du seuil
        thresh_A = np.percentile(img_A, star_percentile)
        thresh_B = np.percentile(img_B, star_percentile)
        clean_A = np.clip(img_A - thresh_A, 0, None)
        clean_B = np.clip(img_B - thresh_B, 0, None)

        shift, error, _ = phase_cross_correlation(
            clean_A, clean_B,
            upsample_factor=upsample_factor,
            normalization=None   # None = plus robuste sur images astronomiques
        )
        return shift   # array [dy, dx]
        
    
    def _debayer_manual(self, img, pattern='RGGB'):
        """Dématriçage manuel optimisé (évite OpenCV)."""
        h, w = img.shape
        rgb = np.zeros((h, w, 3), dtype=float)

        # 1. Extraction simple selon le pattern RGGB (Standard NINA/ZWO)
        if pattern == 'RGGB':
            # R  G
            # G  B
            rgb[0::2, 0::2, 0] = img[0::2, 0::2]       # Rouge
            rgb[1::2, 1::2, 2] = img[1::2, 1::2]       # Bleu
            rgb[0::2, 1::2, 1] = img[0::2, 1::2]       # Vert 1
            rgb[1::2, 0::2, 1] = img[1::2, 0::2]       # Vert 2
        
        # 2. Interpolation pour boucher les trous (Moyenne des voisins)
        k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) * 0.25
        
        # Rouge & Bleu (Trous remplis par les 4 voisins)
        r, b = rgb[:,:,0], rgb[:,:,2]
        r_fill = ndimage.convolve(r, k) * 4
        b_fill = ndimage.convolve(b, k) * 4
        rgb[:,:,0] = r + r_fill * (r == 0)
        rgb[:,:,2] = b + b_fill * (b == 0)
        
        # Vert (Structure en quinconce)
        g = rgb[:,:,1]
        mask_missing_g = (rgb[:,:,0] > 0) | (rgb[:,:,2] > 0)
        g_fill = ndimage.convolve(g, k) * 4
        rgb[:,:,1] = g + (g_fill * mask_missing_g)

        return rgb

    def _smart_load(self, path, is_master=False):
        """
        Charge TIFF, PNG ou FITS et renvoie une image normalisée (0-1).
        Gère le dématriçage automatique pour les FITS couleurs.
        """
        ext = os.path.splitext(path)[1].lower()
        data = None

        # A. Chargement FITS
        if ext in ['.fits', '.fit']:
            with fits.open(path) as hdul:
                raw = hdul[0].data.astype(float)
                # Normalisation 16 bits -> 0-1
                if np.max(raw) > 1.0: raw /= 65535.0
                
                # FITS est souvent tête en bas par rapport aux PNG
                raw = np.flipud(raw)
                
                # Si c'est une image brute (2D) et pas un master (Dark/Flat), on dématrice
                if raw.ndim == 2 and not is_master:
                    # On essaie de lire le pattern ou on suppose RGGB
                    pattern = hdul[0].header.get('BAYERPAT', 'RGGB').replace("'", "").strip()
                    if len(pattern) != 4: pattern = 'RGGB'
                    data = self._debayer_manual(raw, pattern)
                else:
                    data = raw

        # B. Chargement Classique (Imageio)
        else:
            raw = imageio.imread(path).astype(float)
            # Normalisation Auto (8bits ou 16bits)
            limit = 65535.0 if np.max(raw) > 255 else 255.0
            data = raw / limit

        return data
        
    def load_files(self, file_paths):
        if not file_paths: return
        
        # 1. Masters
        self.dark, self.flat = 0, 1
        base_dir = os.path.dirname(file_paths[0])
        for f in glob.glob(os.path.join(base_dir, "master*")):
            try:
                norm = self._smart_load(f, is_master=True)
                if "dark" in os.path.basename(f).lower(): self.dark = norm
                if "flat" in os.path.basename(f).lower(): self.flat = norm / (np.mean(norm)+1e-10)
            except: pass

        # 2. Empilement
        all_aligned = []
        ref_full = None
        target_shape = None
        
        # On trie pour être sûr que la référence est toujours la même
        file_paths = sorted(file_paths)
        
        for i, p in enumerate(file_paths):
            if "master" in os.path.basename(p).lower(): continue
            
            try:
                raw = self._smart_load(p, is_master=False)
                calib = (raw - self.dark) / (self.flat + 1e-6)
                
                # --- PREMIERE IMAGE (REF) ---
                if ref_full is None:
                    target_shape = calib.shape
                    ref_full = calib
                    all_aligned.append(calib)
                    self.log(f"Image {i} : is the reference img") 
                    continue
                
                # --- ALIGNEMENT ---
                if calib.shape != target_shape:
                    if calib.shape[0] == target_shape[1]: calib = np.rot90(calib)
                    else: continue
                
                self.log(f"Aligning {i+1}/{len(file_paths)}...")

                # Extraction du canal Vert (ou Mono) pour l'alignement
                img_A = ref_full[:,:,1] if ref_full.ndim == 3 else ref_full
                img_B = calib[:,:,1]    if calib.ndim == 3    else calib

                # Calcul du décalage par phase cross-correlation
                shift = self.get_shift(img_A, img_B)
                self.log(f"Image {i}: shift detected dy={shift[0]:.1f}, dx={shift[1]:.1f}")

                # Sécurité : shift délirant (> 200 pixels) → image rejetée
                if abs(shift[0]) > 200 or abs(shift[1]) > 200:
                    self.log(f"Image {i}: rejected (shift too large: {shift[0]:.0f}, {shift[1]:.0f})")
                    continue

                # --- STACKING DES COULEURS ---
                if calib.ndim == 3:
                    aligned = np.stack([ndimage.shift(calib[:,:,c], shift, order=1) for c in range(3)], axis=-1)
                else:
                    aligned = ndimage.shift(calib, shift, order=1)
                    
                all_aligned.append(aligned)
                
            except Exception as e:
                print(f"Error {p}: {e}")
        
        # 3. FINALISATION
        if all_aligned:
            self.log(f"Stacking all {len(all_aligned)} images...")
            stacked = np.sum(all_aligned, axis=0)
            
            # Black Point Auto (Pedestal)
            bp_pc = 1
            if stacked.ndim == 3:
                for c in range(3): stacked[:,:,c] -= np.percentile(stacked[:,:,c], bp_pc)
            else:
                stacked -= np.percentile(stacked, bp_pc)

            # White Point Auto
            wp_pc = 99.9
            p_high = np.percentile(stacked, wp_pc)
            self.calibrated = np.clip(stacked / (p_high + 1e-10), 0, 1)
            
            if self.calibrated.ndim == 2: self.calibrated = self.calibrated[:,:,None]
            self.log("Ready")
            
                        
    def remove_gradient(self, img):
        self.log(f"Removing gradient...")

        try:
            h, w = img.shape[:2]
            out = img.copy()
            
            # Paramètres de la grille (8x8 zones)
            grid_size = 8
            step_y, step_x = h // grid_size, w // grid_size
            
            for i in range(img.shape[2]):
                chan = img[:,:,i]
                # Création d'une carte de fond basse résolution
                bg_map_small = np.zeros((grid_size, grid_size))
                
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        # Extraction d'une zone (tile)
                        tile = chan[gy*step_y:(gy+1)*step_y, gx*step_x:(gx+1)*step_x]
                        # On prend le 10ème percentile pour éviter de mesurer la galaxie ou les étoiles
                        bg_map_small[gy, gx] = np.percentile(tile, 10)
                
                # On ré-étire cette carte à la taille d'origine avec un lissage cubique
                # pour créer un modèle de gradient très doux
                bg_model = ndimage.zoom(bg_map_small, (h/grid_size, w/grid_size), order=3)
                bg_model = bg_model[:h, :w] # Recalage dimensions
                
                # Soustraction souple
                out[:,:,i] = np.clip(chan - bg_model, 0, 1)
            
            # Rééquilibrage final pour éviter les zones "trouées"
            return out - np.min(out)
        except Exception as e:
            return img

    
    # --- 1. GESTION DES MASQUES (V3 : Morphologie & Protection) ---

    def _make_star_mask_v3_dilated(self, gray_img, sensitivity=3.0):
        """Détecte les étoiles et dilate la sélection pour couvrir les halos."""
        # High Pass pour isoler les pics
        low_freq = ndimage.gaussian_filter(gray_img, sigma=8)
        high_freq = gray_img - low_freq
        
        # Seuillage
        limit = np.std(high_freq) * sensitivity
        binary_mask = (high_freq > limit)
        
        # Dilation
        return ndimage.binary_dilation(binary_mask, iterations=2).astype(float)

    def _make_galaxy_mask_v3_morpho(self, gray_img, galaxy_sigma):
        """Isole la nébuleuse en supprimant d'abord les étoiles"""
        if galaxy_sigma <= 0.1: return np.zeros_like(gray_img)

        # Suppression des étoiles par morphologie (carré 5x5)
        starless = ndimage.grey_opening(gray_img, size=(5,5))
        smooth_starless = ndimage.gaussian_filter(starless, sigma=4)
        
        # Seuillage adaptatif (Moyenne + 0.5 Sigma)
        bg_mean = np.mean(smooth_starless)
        bg_std = np.std(smooth_starless)
        threshold = bg_mean + (0.5 * bg_std)
        
        return (smooth_starless > threshold).astype(float)

    def extract_masks(self, img, stars_sigma=15, galaxy_sigma=20):
        """Coordonne la création des masques"""
        gray = np.mean(img, axis=2)
        
        # 1. Masque Etoiles
        raw_star = self._make_star_mask_v3_dilated(gray, sensitivity=3.0)
        s_mask = ndimage.gaussian_filter(raw_star, sigma=stars_sigma)
        
        # 2. Masque Galaxie
        raw_gal = self._make_galaxy_mask_v3_morpho(gray, galaxy_sigma)
        
        # 3. Protection : On soustrait les étoiles gonflées du masque galaxie
        if np.max(raw_gal) > 0:
            star_shield = ndimage.binary_dilation(raw_star, iterations=4).astype(float)
            raw_gal = np.clip(raw_gal - star_shield, 0, 1)
            
        g_mask = ndimage.gaussian_filter(raw_gal, sigma=galaxy_sigma)
        
        # 4. Stockage
        self.masks = {
            'stars': np.clip(s_mask, 0, 1),
            'galaxy': np.clip(g_mask, 0, 1),
            'background': np.clip(1.0 - g_mask - s_mask, 0, 1)
        }

    def _step_denoise_harmonized(self, img, amount, galaxy_sigma):
        """Denoise RGB """
        if amount <= 0: return img
        
        # Sigma (y,x,0) pour ne pas mélanger les couleurs
        img_strong = ndimage.gaussian_filter(img, sigma=(amount, amount, 0))
        
        if galaxy_sigma > 0.0 and 'background' in self.masks:
            # Objet lissé à 40% seulement
            img_light = ndimage.gaussian_filter(img, sigma=(amount * 0.4, amount * 0.4, 0))
            m_back = self.masks['background'][..., None]
            return (img_strong * m_back) + (img_light * (1.0 - m_back))
        else:
            return img_strong

    def _step_clarity_multiscale(self, img, strength, galaxy_sigma):
        """Clarity Volume : Mélange détails fins et moyens."""
        if strength <= 0: return img
        
        fine = img - ndimage.gaussian_filter(img, sigma=(2, 2, 0))
        medium = img - ndimage.gaussian_filter(img, sigma=(8, 8, 0))
        structure = (fine * 0.5) + (medium * 1.5)
        
        if galaxy_sigma > 0.1 and 'galaxy' in self.masks:
            return img + (structure * strength * self.masks['galaxy'][..., None])
        else:
            return img + (structure * strength)

    def _step_dynamic_compression(self, img, bp_strength, galaxy_sigma):
        """
        Compression Dynamique (pour le Black Point).
        - Ancrage du noir (soustraction du plancher de bruit).
        - Gamma sélectif sur le fond.
        """
        if self.calibrated is None: return img
        
        # Sécurité
        img = np.clip(img, 0, 1)
        
        # Ancrage (Auto-Pedestal)
        if galaxy_sigma > 0.1 and 'background' in self.masks:
             luma = np.mean(img, axis=2)
             bg_vals = luma[self.masks['background'] > 0.5]
             if len(bg_vals) > 0:
                 pedestal = np.percentile(bg_vals, 5) # On enlève le socle
                 img = np.clip(img - pedestal, 0, 1)

        # Gamma
        gamma_strength = 1.0 + ((bp_strength - 1.0) * 2.0)
        
        if galaxy_sigma > 0.1 and 'background' in self.masks:
            gamma_map = 1.0 + ((gamma_strength - 1.0) * self.masks['background'])
            gamma_map = gamma_map[..., None]
            return np.power(img, gamma_map)
        else:
            return np.power(img, gamma_strength)

    def _step_saturation_smart(self, img, sat_amount, galaxy_sigma):
        """Saturation HSV sélective (protège le fond)."""
        if sat_amount == 1.0: return img
        
        hsv = rgb_to_hsv(np.clip(img, 0, 1))
        
        if galaxy_sigma > 0.0 and 'galaxy' in self.masks:
            # Fond -> 0.5, Objet -> sat_amount
            target_sat = 0.5 + (sat_amount - 0.5) * self.masks['galaxy']
            hsv[:,:,1] *= target_sat
        else:
            hsv[:,:,1] *= sat_amount
            
        return hsv_to_rgb(np.clip(hsv, 0, 1))

    # =========================================================================
    # FONCTION PRINCIPALE
    # =========================================================================

    def apply_cosmetics(self, stretch, bp_ratio, clarity, denoise, saturation, galaxy_sigma, stars_sigma, rgb=(1,1,1), do_grad=True):
        if self.calibrated is None: return None
        img = self.calibrated.copy()
        
        # 1. Extraction Masques
        self.log(f"Generating masks...")
        self.extract_masks(img, stars_sigma=stars_sigma, galaxy_sigma=galaxy_sigma)
        
        # 2. Gradient & Balance RGB
        if do_grad and hasattr(self, 'remove_gradient'): 
            self.log(f"Removing gradient...")
            img = self.remove_gradient(img)
            
        self.log(f"Balancing colors...")
        for i in range(3): img[:,:,i] *= rgb[i]

        # A. Denoise (Texture propre)
        if denoise > 0:
            self.log(f"Denoise...")
            img = self._step_denoise_harmonized(img, denoise, galaxy_sigma)
            
        # B. Clarity (Volume & Détails)
        if clarity > 0:
            self.log(f"Clarity...")
            img = self._step_clarity_multiscale(img, clarity, galaxy_sigma)
            
        # C. Compression (Gestion du fond noir & Contraste)
        if bp_ratio != 1.0:
            self.log(f"Black point...")
            img = self._step_dynamic_compression(img, bp_ratio, galaxy_sigma)
            
        # D. Saturation (Couleur finale)
        if saturation > 0:
            self.log(f"Color saturation...")
            img = self._step_saturation_smart(img, saturation, galaxy_sigma)

        # 4. Stretch Final
        if stretch > 0.01: 
            self.log(f"Stretching...")
            img = np.clip(AsinhStretch(a=stretch)(img), 0, 1)

        return img

        