# ==============================================================
# ENHANCED WAVE DETECTION + DIFFRACTION / REFRACTION ANALYSIS
# ==============================================================
# Versión robusta: maneja casos con pocos puntos.
# ==============================================================

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import glob
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
CSV_PATH = os.path.join(OUTPUT_DIR, "wave_parameters.csv")
RGB_PATH = glob.glob(os.path.join(BASE_DIR, "*ortho_visual.tif"))
if not RGB_PATH:
    raise FileNotFoundError("No se encontró ortho_visual.tif")
RGB_PATH = RGB_PATH[0]

# Parámetros
VARIANCE_RADIUS = 5
GRADIENT_RADIUS = 5
TH_VARIANCE = 400
TH_CURVATURE = 0.1

# ==============================================================
# CARGA DE DATOS
# ==============================================================

# 1. Imagen RGB
with rasterio.open(RGB_PATH) as src:
    r = src.read(1).astype(np.float32)
    g = src.read(2).astype(np.float32)
    b = src.read(3).astype(np.float32)
    height, width = r.shape
    transform = src.transform

def norm_pct(img):
    v = img[np.isfinite(img)]
    if len(v) == 0: return np.zeros_like(img)
    vmin, vmax = np.percentile(v, [2, 98])
    if vmax == vmin: return np.zeros_like(img)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

rgb = np.stack([norm_pct(r), norm_pct(g), norm_pct(b)], axis=-1)
print(f"RGB cargado: {rgb.shape}")

# 2. Máscara de agua (clase 0)
with rasterio.open(CLASSIFICATION_PATH) as src:
    class_map = src.read(1).astype(np.int16)
water_mask = (class_map == 0)
print(f"Máscara de agua: {np.sum(water_mask):,} píxeles")

# 3. Mapa de probabilidad
if os.path.exists(PROB_MAP_PATH):
    with rasterio.open(PROB_MAP_PATH) as src:
        prob_map = src.read(1).astype(np.float32)
else:
    prob_map = np.zeros((height, width), dtype=np.float32)
    print("Mapa de probabilidad no encontrado. Se usará cero.")

# 4. Parámetros de ondas (CSV)
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"El archivo {CSV_PATH} no existe. Ejecuta primero wave_detection_water.py.")

df = pd.read_csv(CSV_PATH)
print(f"CSV cargado: {len(df)} registros")

# ---- IMPORTANTE: NO FILTRAMOS POR PROBABILIDAD ----
df = df[df['wavelength_m'] > 0]   # solo longitudes válidas
print(f"Registros con longitud > 0: {len(df)}")

if len(df) == 0:
    raise ValueError("No hay registros con longitud de onda válida (wavelength_m > 0).")

# ==============================================================
# FUNCIONES DE MÉTRICAS (con manejo de pocos puntos)
# ==============================================================

def local_orientation_gradient(df, radius=5, min_neighbors=2):
    coords = df[['row', 'col']].values
    if len(coords) < min_neighbors:
        return np.zeros(len(coords))
    orients = np.deg2rad(df['orientation_deg'].values)
    gradients = []
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    for i, (r, c) in enumerate(coords):
        indices = nbrs.radius_neighbors([(r, c)], radius=radius, return_distance=False)[0]
        if len(indices) < min_neighbors:
            gradients.append(0)
            continue
        neigh_orients = orients[indices]
        mean_cos = np.mean(np.cos(neigh_orients))
        mean_sin = np.mean(np.sin(neigh_orients))
        mean_orient = np.arctan2(mean_sin, mean_cos)
        diffs = np.arctan2(np.sin(neigh_orients - mean_orient), np.cos(neigh_orients - mean_orient))
        gradients.append(np.std(diffs))
    return np.array(gradients)

def local_curvature(df, radius=5, min_neighbors=2):
    coords = df[['row', 'col']].values
    if len(coords) < min_neighbors:
        return np.zeros(len(coords))
    orients = np.deg2rad(df['orientation_deg'].values)
    curvatures = []
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    for i, (r, c) in enumerate(coords):
        indices = nbrs.radius_neighbors([(r, c)], radius=radius, return_distance=False)[0]
        if len(indices) < min_neighbors:
            curvatures.append(0)
            continue
        neigh_coords = coords[indices]
        neigh_orients = orients[indices]
        mean_cos = np.mean(np.cos(neigh_orients))
        mean_sin = np.mean(np.sin(neigh_orients))
        mean_orient = np.arctan2(mean_sin, mean_cos)
        u = np.sin(mean_orient)
        v = -np.cos(mean_orient)
        diffs = neigh_coords - np.array([r, c])
        proj = diffs[:, 0] * u + diffs[:, 1] * v
        orient_diffs = np.arctan2(np.sin(neigh_orients - mean_orient), np.cos(neigh_orients - mean_orient))
        if np.std(proj) > 1e-6:
            slope = np.polyfit(proj, orient_diffs, 1)[0]
            curvatures.append(np.abs(slope))
        else:
            curvatures.append(0)
    return np.array(curvatures)

def local_variance_circular(df, radius=5, min_neighbors=2):
    coords = df[['row', 'col']].values
    if len(coords) < min_neighbors:
        return np.zeros(len(coords))
    orients = np.deg2rad(df['orientation_deg'].values)
    variances = []
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    for i, (r, c) in enumerate(coords):
        indices = nbrs.radius_neighbors([(r, c)], radius=radius, return_distance=False)[0]
        if len(indices) < min_neighbors:
            variances.append(0)
            continue
        neigh_orients = orients[indices]
        mean_vector = np.mean(np.exp(1j * neigh_orients))
        var = 1 - np.abs(mean_vector)
        variances.append(var * (180**2))
    return np.array(variances)

# ==============================================================
# CALCULAR MÉTRICAS (con fallback si hay pocos puntos)
# ==============================================================

print("Calculando métricas locales...")
if len(df) >= 2:
    df['gradient'] = local_orientation_gradient(df, radius=GRADIENT_RADIUS)
    df['curvature'] = local_curvature(df, radius=GRADIENT_RADIUS)
    df['variance'] = local_variance_circular(df, radius=VARIANCE_RADIUS)
else:
    print("ADVERTENCIA: Solo un punto. Las métricas de vecindad se fijan a 0.")
    df['gradient'] = 0
    df['curvature'] = 0
    df['variance'] = 0

# Normalizar
max_grad = np.percentile(df['gradient'], 95) if np.max(df['gradient']) > 0 else 1
df['gradient_norm'] = np.clip(df['gradient'] / max_grad, 0, 1)
max_curv = np.percentile(df['curvature'], 95) if np.max(df['curvature']) > 0 else 1
df['curvature_norm'] = np.clip(df['curvature'] / max_curv, 0, 1)

# ==============================================================
# GENERAR MAPAS (si hay suficientes puntos, sino crear mapas vacíos)
# ==============================================================

def create_map_from_df(df, column, height, width, default=0):
    map_array = np.full((height, width), default, dtype=np.float32)
    for idx, row in df.iterrows():
        r, c = int(row['row']), int(row['col'])
        if 0 <= r < height and 0 <= c < width:
            map_array[r, c] = row[column]
    return gaussian_filter(map_array, sigma=2)

if len(df) >= 1:
    grad_map = create_map_from_df(df, 'gradient_norm', height, width)
    curv_map = create_map_from_df(df, 'curvature_norm', height, width)
    var_map = create_map_from_df(df, 'variance', height, width)
else:
    grad_map = np.zeros((height, width), dtype=np.float32)
    curv_map = np.zeros((height, width), dtype=np.float32)
    var_map = np.zeros((height, width), dtype=np.float32)

prof = {"driver":"GTiff", "dtype":"float32", "width":width, "height":height,
        "count":1, "crs":None, "transform":transform, "compress":"lzw", "nodata":0}
with rasterio.open(os.path.join(OUTPUT_DIR, "refraction_gradient.tif"), "w", **prof) as dst:
    dst.write(grad_map, 1)
with rasterio.open(os.path.join(OUTPUT_DIR, "diffraction_curvature.tif"), "w", **prof) as dst:
    dst.write(curv_map, 1)
with rasterio.open(os.path.join(OUTPUT_DIR, "interference_variance.tif"), "w", **prof) as dst:
    dst.write(var_map, 1)

print("Mapas guardados.")

# ==============================================================
# VISUALIZACIÓN (robusta)
# ==============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
ax1, ax2, ax3, ax4 = axes.ravel()

# 1. RGB + agua
ax1.imshow(rgb, origin='upper')
ax1.imshow(water_mask, cmap='Blues', alpha=0.2, origin='upper')
ax1.set_title("RGB + Water Mask", fontsize=14)
ax1.axis('off')

# 2. Mapa de probabilidad
im2 = ax2.imshow(prob_map, cmap='hot', vmin=0, vmax=1, origin='upper')
ax2.set_title("Wave Probability", fontsize=14)
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046)

# 3. Orientación + refracción (si hay puntos)
ax3.imshow(rgb, origin='upper', alpha=0.7)
if len(df) > 0:
    for idx, row in df.iterrows():
        r, c = row['row'], row['col']
        orient = row['orientation_deg']
        theta = np.deg2rad(orient)
        u = np.sin(theta)
        v = -np.cos(theta)
        scale = row['wavelength_m'] / 30
        scale = np.clip(scale, 5, 40)
        color = plt.cm.RdYlGn_r(row['gradient_norm'])
        ax3.quiver(c, r, u, v, angles='xy', scale_units='xy', scale=1/scale,
                   color=color, width=0.003, headwidth=3, headlength=4, alpha=0.9)
ax3.set_title("Wave Direction (color = refraction gradient)", fontsize=14)
ax3.axis('off')

# 4. Difracción + interferencia
ax4.imshow(rgb, origin='upper', alpha=0.6)
if len(df) > 0:
    if np.sum(df['curvature_norm'] > TH_CURVATURE) > 0:
        diffr = df[df['curvature_norm'] > TH_CURVATURE]
        ax4.scatter(diffr['col'], diffr['row'], c='yellow', s=30, edgecolor='black', linewidth=0.5, label='Diffraction')
    if np.sum(df['variance'] > TH_VARIANCE) > 0:
        interf = df[df['variance'] > TH_VARIANCE]
        ax4.scatter(interf['col'], interf['row'], c='red', s=50, marker='x', label='Interference')
ax4.set_title("Diffraction & Interference Zones", fontsize=14)
ax4.axis('off')
ax4.legend(loc='lower right', fontsize=10)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "wave_diffraction_refraction.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figura guardada en: {fig_path}")

# ==============================================================
# GUARDAR RESULTADOS
# ==============================================================

df.to_csv(os.path.join(OUTPUT_DIR, "wave_parameters_with_metrics.csv"), index=False)
print("Datos con métricas guardados en: wave_parameters_with_metrics.csv")

print("\nAnálisis completado.")