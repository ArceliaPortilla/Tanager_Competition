# ==============================================================
# PLOT WAVE VECTORS ON WATER MASK + LINEAR WAVE THEORY
# ==============================================================
# Superpone flechas de dirección y longitud de onda sobre la
# imagen RGB original (solo en la zona de agua).
# Aplica teoría lineal de ondas (relación de dispersión).
# Detecta zonas de difracción/interferencia (alta varianza de orientación).
# ==============================================================

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
import glob
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURACIÓN
# ==============================================================

BASE_DIR = r"D:\Arcelia\Tangent"
OUTPUT_DIR = os.path.join(BASE_DIR, "wave_detection_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "wave_parameters.csv")
CLASSIFICATION_PATH = os.path.join(OUTPUT_DIR, "random_forest_classification.tif")
RGB_PATH = glob.glob(os.path.join(BASE_DIR, "*ortho_visual.tif"))
if not RGB_PATH:
    raise FileNotFoundError("No se encontró ortho_visual.tif")
RGB_PATH = RGB_PATH[0]

# Parámetros de visualización
MAX_ARROWS = 300
ARROW_SCALE = 0.5
COLOR_BY = 'wavelength_m'       # 'wavelength_m' o 'energy_ratio' o 'probability'
WINDOW_VARIANCE = 5
TH_VARIANCE = 500

# Constantes físicas
G = 9.81

# ==============================================================
# CARGA DE DATOS
# ==============================================================

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

with rasterio.open(CLASSIFICATION_PATH) as src:
    class_map = src.read(1).astype(np.int16)
water_mask = (class_map == 0)
print(f"Máscara de agua: {np.sum(water_mask):,} píxeles")

df = pd.read_csv(CSV_PATH)
print(f"CSV cargado: {len(df)} registros")
df = df[df['wavelength_m'] > 0]
print(f"Registros con longitud válida: {len(df)}")

if len(df) > MAX_ARROWS:
    np.random.seed(42)
    df_sample = df.sample(n=MAX_ARROWS, random_state=42)
else:
    df_sample = df
print(f"Muestras a graficar: {len(df_sample)}")

# ==============================================================
# CALCULAR VARIANZA LOCAL
# ==============================================================

def local_variance_orientation(df, radius=5):
    variances = []
    coords = df[['row', 'col']].values
    orients = np.deg2rad(df['orientation_deg'].values)
    for i, (r, c) in enumerate(coords):
        dist = np.sqrt((coords[:, 0] - r)**2 + (coords[:, 1] - c)**2)
        neighbors = dist <= radius
        if np.sum(neighbors) < 3:
            variances.append(0)
            continue
        neigh_orient = orients[neighbors]
        mean_vector = np.mean(np.exp(1j * neigh_orient))
        circular_variance = 1 - np.abs(mean_vector)
        variances.append(circular_variance * (180**2))
    return np.array(variances)

df_sample['variance'] = local_variance_orientation(df, radius=WINDOW_VARIANCE)
diffraction_mask = df_sample['variance'] > TH_VARIANCE
print(f"Puntos con alta varianza: {np.sum(diffraction_mask)}")

# ==============================================================
# TEORÍA LINEAL
# ==============================================================

df_sample['period_sec'] = np.sqrt((2 * np.pi * df_sample['wavelength_m']) / G)
print(f"Periodo medio estimado: {df_sample['period_sec'].mean():.2f} s")

# ==============================================================
# GRÁFICO PRINCIPAL
# ==============================================================

fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(rgb, origin='upper')
ax.set_xlim(0, width)
ax.set_ylim(height, 0)

if COLOR_BY == 'wavelength_m':
    norm = plt.Normalize(vmin=df_sample['wavelength_m'].min(), vmax=df_sample['wavelength_m'].max())
    cmap = plt.cm.viridis
    color_label = 'Wavelength (m)'
elif COLOR_BY == 'energy_ratio':
    norm = plt.Normalize(vmin=df_sample['energy_ratio'].min(), vmax=df_sample['energy_ratio'].max())
    cmap = plt.cm.plasma
    color_label = 'Energy Ratio'
else:
    norm = plt.Normalize(vmin=df_sample['probability'].min(), vmax=df_sample['probability'].max())
    cmap = plt.cm.hot
    color_label = 'Probability'

for idx, row in df_sample.iterrows():
    r, c = row['row'], row['col']
    orient_deg = row['orientation_deg']
    wavelength = row['wavelength_m']
    color_val = row[COLOR_BY]
    theta_rad = np.deg2rad(orient_deg)
    u = np.sin(theta_rad)
    v = -np.cos(theta_rad)
    scale = wavelength / 20
    if scale < 5: scale = 5
    if scale > 50: scale = 50
    ax.quiver(c, r, u, v, angles='xy', scale_units='xy', scale=1/scale,
              color=cmap(norm(color_val)), width=0.003, headwidth=3, headlength=4, alpha=0.8)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label(color_label, fontsize=12)

# Marcas de difracción/interferencia
if np.sum(diffraction_mask) > 0:
    for idx, row in df_sample[diffraction_mask].iterrows():
        r, c = row['row'], row['col']
        radius_pixels = row['wavelength_m'] / 30
        if radius_pixels < 5: radius_pixels = 10
        if radius_pixels > 80: radius_pixels = 80
        circle = Circle((c, r), radius=radius_pixels, edgecolor='red', facecolor='none', linewidth=2, alpha=0.8)
        ax.add_patch(circle)
    ax.scatter(df_sample[diffraction_mask]['col'], df_sample[diffraction_mask]['row'],
               s=80, c='red', marker='x', label='Possible diffraction/interference', zorder=5)

# Texto teoría lineal
mean_period = df_sample['period_sec'].mean()
mean_wavelength = df_sample['wavelength_m'].mean()
text_str = (f"Linear Wave Theory (Deep Water)\n"
            f"Mean Wavelength: {mean_wavelength:.1f} m\n"
            f"Mean Period (T): {mean_period:.2f} s\n"
            f"Diffraction zones: {np.sum(diffraction_mask)}")
ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_title('Wave Vectors on Water (Direction & Wavelength)\nRed circles = high orientation variance (diffraction/interference)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Column (pixels)', fontsize=12)
ax.set_ylabel('Row (pixels)', fontsize=12)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='none', edgecolor='red', linewidth=2, 
                         label='Diffraction/Interference zone')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'wave_vectors_overlay.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figura guardada en: {output_path}")

# Guardar resultados teoría lineal
df_output = df_sample[['row', 'col', 'orientation_deg', 'wavelength_m', 
                       'energy_ratio', 'probability', 'period_sec', 'variance']]
df_output.to_csv(os.path.join(OUTPUT_DIR, 'wave_linear_theory_results.csv'), index=False)
print("Resultados de teoría lineal guardados en: wave_linear_theory_results.csv")

print("\nAnálisis completado.")