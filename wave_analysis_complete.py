# ==============================================================
# WAVE DETECTION + ENHANCEMENT + PARAMETER EXTRACTION (UNIFIED)
# ==============================================================
# Este script unificado:
#   1. Carga la clasificación (máscara de agua) y las bandas espectrales.
#   2. Detecta ondas en agua usando Wavelet + FFT (genera mapa de probabilidad).
#   3. Realza la imagen en zonas con ondas.
#   4. Extrae parámetros físicos (orientación, longitud, etc.) y guarda CSV.
#   5. Genera mapas de orientación y longitud de onda.
#   6. Muestra figura resumen y guarda resultados.
# ==============================================================

import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
from scipy.ndimage import gaussian_filter, laplace, sobel
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURACIÓN
# ==============================================================

BASE_DIR = r"D:\Arcelia\Tangent"
OUTPUT_DIR = os.path.join(BASE_DIR, "wave_detection_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSIFICATION_PATH = os.path.join(OUTPUT_DIR, "random_forest_classification.tif")
PROB_MAP_PATH = os.path.join(OUTPUT_DIR, "wave_probability_map.tif")
ORIENT_MAP_PATH = os.path.join(OUTPUT_DIR, "wave_orientation.tif")
WAVELENGTH_MAP_PATH = os.path.join(OUTPUT_DIR, "wave_wavelength.tif")
CSV_PATH = os.path.join(OUTPUT_DIR, "wave_parameters.csv")
ENHANCED_RGB_PATH = os.path.join(OUTPUT_DIR, "enhanced_waves_rgb.tif")
SUMMARY_FIG_PATH = os.path.join(OUTPUT_DIR, "wave_parameters_summary.png")

# Parámetros de detección
WINDOW_SIZE = 64
MIN_WATER_PIXELS = 0.6
FREQ_BANDS = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.5)]
N_SAMPLES = 3000                # Número de píxeles de agua a muestrear para detección
WAVELET_NAME = 'db4'
USE_BAND = 'green'              # 'blue', 'green' o 'red'
TH_PROB = 0.6                   # Umbral de probabilidad para realce (se ajusta automáticamente)

# Parámetros de extracción (deben coincidir con detección)
PIXEL_SIZE = 30.0               # metros (Tanager ~30m)
SAMPLE_SIZE_PARAMS = 200        # Número de puntos para extraer parámetros

# ==============================================================
# CARGA DE DATOS
# ==============================================================

# 1. Cargar clasificación
with rasterio.open(CLASSIFICATION_PATH) as src:
    class_map = src.read(1).astype(np.int16)
    transform = src.transform
    crs = src.crs
    height, width = class_map.shape

water_mask = (class_map == 0)
print(f"Máscara de agua: {np.sum(water_mask):,} píxeles")

# 2. Cargar imagen RGB (ortho_visual.tif)
rgb_files = glob.glob(os.path.join(BASE_DIR, "*ortho_visual.tif")) + \
            glob.glob(os.path.join(BASE_DIR, "*ortho_visual.tiff"))
if rgb_files:
    with rasterio.open(rgb_files[0]) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
    def norm_pct(img):
        v = img[np.isfinite(img)]
        if len(v)==0: return np.zeros_like(img)
        vmin, vmax = np.percentile(v, [2,98])
        if vmax==vmin: return np.zeros_like(img)
        return np.clip((img-vmin)/(vmax-vmin), 0, 1)
    rgb_display = np.stack([norm_pct(r), norm_pct(g), norm_pct(b)], axis=-1)
else:
    raise FileNotFoundError("No se encontró ortho_visual.tif")

print(f"RGB cargado: {rgb_display.shape}")

# 3. Cargar bandas espectrales desde HDF5
import h5py
sr_files = glob.glob(os.path.join(BASE_DIR, "*ortho_sr_hdf5.h5")) + \
           glob.glob(os.path.join(BASE_DIR, "*ortho_sr_hdf5.hdf"))
if not sr_files:
    raise FileNotFoundError("No se encontró HDF5 de reflectancia.")
SR_HDF5 = sr_files[0]

with h5py.File(SR_HDF5, 'r') as f:
    def find_sr(name, obj):
        if isinstance(obj, h5py.Dataset) and ('surface_reflectance' in name or 'radiance' in name):
            return name
        return None
    ds_name = f.visititems(find_sr)
    if ds_name is None:
        raise ValueError("No se encontró dataset de reflectancia.")
    cube = f[ds_name][:].astype(np.float32)
    def find_wl(name, obj):
        if isinstance(obj, h5py.Dataset) and 'wavelength' in name.lower():
            return name
        return None
    wl_name = f.visititems(find_wl)
    wavelengths = f[wl_name][:] if wl_name is not None else np.linspace(400, 2500, cube.shape[0])
    # Extraer banda seleccionada
    target_wl = {'blue':490, 'green':560, 'red':665}
    idx_band = np.argmin(np.abs(wavelengths - target_wl[USE_BAND]))
    band_image = cube[idx_band, :, :]

# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================

def extract_window(image, r, c, size):
    half = size // 2
    r0, r1 = max(0, r-half), min(height, r+half)
    c0, c1 = max(0, c-half), min(width, c+half)
    win = image[r0:r1, c0:c1]
    if win.shape != (size, size):
        tmp = np.zeros((size, size), dtype=win.dtype)
        tmp[:win.shape[0], :win.shape[1]] = win
        win = tmp
    return win

def wavelet_features(img):
    coeffs = pywt.dwt2(img, WAVELET_NAME, mode='symmetric')
    cA, (cH, cV, cD) = coeffs
    energies = [np.sum(c**2) for c in (cA, cH, cV, cD)]
    total = sum(energies)
    if total == 0: return np.zeros(4)
    return np.array(energies) / total

def fft_band_energy(img):
    fft = fftshift(fft2(img))
    energy = np.abs(fft)**2
    rows, cols = img.shape
    f_rows = fftshift(fftfreq(rows, d=1))
    f_cols = fftshift(fftfreq(cols, d=1))
    band_energies = []
    for (fmin, fmax) in FREQ_BANDS:
        mask_row = (np.abs(f_rows) >= fmin) & (np.abs(f_rows) <= fmax)
        mask_col = (np.abs(f_cols) >= fmin) & (np.abs(f_cols) <= fmax)
        mask = np.outer(mask_row, mask_col)
        band_energies.append(np.sum(energy[mask]))
    total = np.sum(energy)
    if total == 0: return np.zeros(len(FREQ_BANDS))
    return np.array(band_energies) / total

def compute_wave_parameters(window, pixel_size=30.0):
    fft = fftshift(fft2(window))
    energy = np.abs(fft)**2
    rows, cols = window.shape
    f_rows = fftshift(fftfreq(rows, d=1))
    f_cols = fftshift(fftfreq(cols, d=1))
    R = np.sqrt(f_rows[:, None]**2 + f_cols[None, :]**2)
    mask = R > 0.02
    total_energy = np.sum(energy)
    if total_energy == 0:
        return {'freq':0, 'wavelength':0, 'orientation':0, 'energy_ratio':0}
    energy_masked = energy * mask
    if np.sum(energy_masked) == 0:
        return {'freq':0, 'wavelength':0, 'orientation':0, 'energy_ratio':0}
    idx_max = np.argmax(energy_masked)
    r_peak, c_peak = np.unravel_index(idx_max, energy_masked.shape)
    freq_r = f_rows[r_peak]
    freq_c = f_cols[c_peak]
    freq_radial = np.sqrt(freq_r**2 + freq_c**2)
    wavelength = 1.0 / freq_radial * pixel_size if freq_radial > 0 else 0
    orientation = np.arctan2(freq_c, freq_r) * 180 / np.pi
    peak_energy = energy[r_peak, c_peak]
    energy_ratio = peak_energy / total_energy
    return {
        'freq_cycles_per_pixel': freq_radial,
        'wavelength_m': wavelength,
        'orientation_deg': orientation,
        'energy_ratio': energy_ratio,
        'peak_row': r_peak,
        'peak_col': c_peak
    }

# ==============================================================
# DETECCIÓN DE ONDAS (si no existe el mapa de probabilidad)
# ==============================================================

if not os.path.exists(PROB_MAP_PATH):
    print("Mapa de probabilidad no encontrado. Generando...")

    # Muestrear píxeles de agua
    water_rows, water_cols = np.where(water_mask)
    if len(water_rows) > N_SAMPLES:
        np.random.seed(42)
        idx = np.random.choice(len(water_rows), N_SAMPLES, replace=False)
        water_rows, water_cols = water_rows[idx], water_cols[idx]

    features = []
    positions = []
    scores = []

    for r, c in zip(water_rows, water_cols):
        win = extract_window(band_image, r, c, WINDOW_SIZE)
        win_mask = extract_window(water_mask, r, c, WINDOW_SIZE)
        if np.mean(win_mask) < MIN_WATER_PIXELS:
            continue
        w_feat = wavelet_features(win)
        fft_feat = fft_band_energy(win)
        feat = np.concatenate([w_feat, fft_feat])
        features.append(feat)
        positions.append((r, c))
        scores.append(fft_feat[1])  # banda media

    if len(features) == 0:
        raise RuntimeError("No hay ventanas con suficiente agua para la detección.")

    features = np.array(features)
    scores = np.array(scores)

    # Generar etiqueta sintética y entrenar RF (para obtener probabilidad)
    threshold = np.percentile(scores, 80)
    labels = (scores > threshold).astype(int)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.3, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(features_scaled)[:, 1]

    # Crear mapa de probabilidad
    prob_map = np.zeros((height, width), dtype=np.float32)
    for (r, c), p in zip(positions, proba):
        prob_map[r, c] = p
    prob_map_smooth = gaussian_filter(prob_map, sigma=2)

    # Guardar mapa
    prof = {"driver":"GTiff", "dtype":"float32", "width":width, "height":height,
            "count":1, "crs":crs, "transform":transform, "compress":"lzw", "nodata":0}
    with rasterio.open(PROB_MAP_PATH, "w", **prof) as dst:
        dst.write(prob_map_smooth, 1)
    print("Mapa de probabilidad generado y guardado en:", PROB_MAP_PATH)
else:
    # Cargar mapa existente
    with rasterio.open(PROB_MAP_PATH) as src:
        prob_map_smooth = src.read(1).astype(np.float32)
    print("Mapa de probabilidad cargado.")

# ==============================================================
# REALCE DE IMAGEN
# ==============================================================

print("Generando realce de ondas...")

# Umbral para realce
high_prob = (prob_map_smooth > TH_PROB) & water_mask
rows_high, cols_high = np.where(high_prob)
if len(rows_high) < 50:
    TH_PROB = np.percentile(prob_map_smooth[water_mask], 90)
    high_prob = (prob_map_smooth > TH_PROB) & water_mask
    rows_high, cols_high = np.where(high_prob)
print(f"Píxeles con alta probabilidad (> {TH_PROB:.2f}): {len(rows_high):,}")

# Laplaciano de la banda
band_norm = norm_pct(band_image)
laplacian = laplace(band_norm, mode='constant')
laplacian_norm = norm_pct(np.abs(laplacian))

# Realce
enhanced_rgb = rgb_display.copy()
mask_enhance = high_prob.astype(float)
mask_enhance = gaussian_filter(mask_enhance, sigma=2)
enhance_factor = 0.5
for c in range(3):
    enhanced_rgb[:,:,c] = rgb_display[:,:,c] + enhance_factor * laplacian_norm * mask_enhance
enhanced_rgb = np.clip(enhanced_rgb, 0, 1)

# Guardar realce
img_enh = (enhanced_rgb * 255).astype(np.uint8)
Image.fromarray(img_enh).save(ENHANCED_RGB_PATH)
print("Imagen realzada guardada en:", ENHANCED_RGB_PATH)

# ==============================================================
# EXTRACCIÓN DE PARÁMETROS (si no existe CSV o se fuerza)
# ==============================================================

if not os.path.exists(CSV_PATH) or True:  # forzar regeneración
    print("Extrayendo parámetros de ondas...")

    # Seleccionar puntos de alta probabilidad
    sample_size = min(SAMPLE_SIZE_PARAMS, len(rows_high))
    if sample_size < 10:
        # Si no hay suficientes, tomar de toda el agua con mayor probabilidad
        water_prob = prob_map_smooth[water_mask]
        if len(water_prob) > 0:
            thresh_auto = np.percentile(water_prob, 90)
            high_prob_auto = (prob_map_smooth > thresh_auto) & water_mask
            rows_high, cols_high = np.where(high_prob_auto)
            sample_size = min(SAMPLE_SIZE_PARAMS, len(rows_high))

    if sample_size < 5:
        raise RuntimeError("No hay suficientes puntos con alta probabilidad para extraer parámetros.")

    np.random.seed(42)
    idx_sample = np.random.choice(len(rows_high), sample_size, replace=False)
    rows_sample = rows_high[idx_sample]
    cols_sample = cols_high[idx_sample]

    params_list = []
    for r, c in zip(rows_sample, cols_sample):
        win = extract_window(band_image, r, c, WINDOW_SIZE)
        params = compute_wave_parameters(win, pixel_size=PIXEL_SIZE)
        params['row'] = r
        params['col'] = c
        params['probability'] = prob_map_smooth[r, c]
        params_list.append(params)

    df = pd.DataFrame(params_list)
    df.to_csv(CSV_PATH, index=False)
    print("Parámetros guardados en:", CSV_PATH)
else:
    df = pd.read_csv(CSV_PATH)
    print("Parámetros cargados desde:", CSV_PATH)

# ==============================================================
# GENERAR MAPAS DE ORIENTACIÓN Y LONGITUD DE ONDA
# ==============================================================

orient_map = np.zeros((height, width), dtype=np.float32)
wavelength_map = np.zeros((height, width), dtype=np.float32)
for idx, row in df.iterrows():
    r, c = int(row['row']), int(row['col'])
    orient_map[r, c] = row['orientation_deg']
    wavelength_map[r, c] = row['wavelength_m']

orient_smooth = gaussian_filter(orient_map, sigma=3)
wavelength_smooth = gaussian_filter(wavelength_map, sigma=3)

prof = {"driver":"GTiff", "dtype":"float32", "width":width, "height":height,
        "count":1, "crs":crs, "transform":transform, "compress":"lzw", "nodata":0}
with rasterio.open(ORIENT_MAP_PATH, "w", **prof) as dst:
    dst.write(orient_smooth, 1)
with rasterio.open(WAVELENGTH_MAP_PATH, "w", **prof) as dst:
    dst.write(wavelength_smooth, 1)
print("Mapas de orientación y longitud guardados.")

# ==============================================================
# FIGURA RESUMEN (2x3)
# ==============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

ax1.imshow(rgb_display)
ax1.set_title("Original RGB")
ax1.axis('off')

ax2.imshow(enhanced_rgb)
ax2.set_title("Enhanced waves")
ax2.axis('off')

ax3.imshow(prob_map_smooth, cmap='hot', vmin=0, vmax=1)
ax3.set_title("Wave probability")
ax3.axis('off')

im4 = ax4.imshow(orient_smooth, cmap='twilight', vmin=-180, vmax=180)
ax4.set_title("Orientation (°)")
ax4.axis('off')
plt.colorbar(im4, ax=ax4, fraction=0.046, ticks=[-180, -90, 0, 90, 180])

im5 = ax5.imshow(wavelength_smooth, cmap='viridis')
ax5.set_title("Wavelength (m)")
ax5.axis('off')
plt.colorbar(im5, ax=ax5, fraction=0.046)

# Espectro de una ventana representativa
if len(df) > 0:
    best_idx = df['probability'].idxmax()
    r, c = int(df.loc[best_idx, 'row']), int(df.loc[best_idx, 'col'])
    win = extract_window(band_image, r, c, WINDOW_SIZE)
    fft = fftshift(fft2(win))
    magnitude = np.log(1 + np.abs(fft))
    ax6.imshow(magnitude, cmap='viridis', origin='upper')
    ax6.set_title("FFT spectrum (example)")
    ax6.axis('off')
else:
    ax6.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax6.transAxes)

plt.tight_layout()
plt.savefig(SUMMARY_FIG_PATH, dpi=200, bbox_inches='tight')
plt.show()
print("Figura resumen guardada en:", SUMMARY_FIG_PATH)

print("\nProceso completado. Todos los resultados están en:", OUTPUT_DIR)