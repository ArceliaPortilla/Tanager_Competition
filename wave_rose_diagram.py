# ==============================================================
# WAVE ROSE DIAGRAM (DIRECTIONAL HISTOGRAM) - CLEAN VERSION
# ==============================================================
# Lee el archivo wave_parameters.csv y genera un gráfico polar
# que muestra la distribución de direcciones de las ondas.
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================
# CONFIGURACIÓN
# ==============================================================

CSV_PATH = r"D:\Arcelia\Tangent\wave_detection_output\wave_parameters.csv"
OUTPUT_DIR = r"D:\Arcelia\Tangent\wave_detection_output"
OUTPUT_FIGURE = os.path.join(OUTPUT_DIR, "wave_rose_diagram.png")

BIN_WIDTH_DEG = 15               # Ancho de cada sector en grados
MIN_PROBABILITY = 0.0            # Sin filtro (0 = todos)
COLOR_MAP = 'plasma'

# ==============================================================
# LECTURA DE DATOS
# ==============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"No se encontró el archivo: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Archivo cargado: {CSV_PATH}")
print(f"Registros totales: {len(df)}")

if MIN_PROBABILITY > 0:
    df = df[df['probability'] >= MIN_PROBABILITY]
    print(f"Registros después de filtrar (prob >= {MIN_PROBABILITY}): {len(df)}")

# Verificar columnas necesarias
required_cols = ['orientation_deg', 'wavelength_m', 'energy_ratio', 'probability']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"El archivo CSV no contiene la columna '{col}'")

# Preparar datos
orient_rad = np.deg2rad(df['orientation_deg'].values)
wavelengths = df['wavelength_m'].values
weights = df['energy_ratio'].values
probs = df['probability'].values

# ==============================================================
# CREACIÓN DEL DIAGRAMA ROSA
# ==============================================================

# Definir bins para la dirección (de 0 a 360 grados)
bins_deg = np.arange(0, 360 + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
bins_rad = np.deg2rad(bins_deg)

# Histograma direccional (ponderado por energía)
hist, _ = np.histogram(orient_rad, bins=bins_rad, weights=weights)
counts, _ = np.histogram(orient_rad, bins=bins_rad)

# Centros de los bins
bin_centers_deg = bins_deg[:-1] + BIN_WIDTH_DEG/2
bin_centers_rad = np.deg2rad(bin_centers_deg)

# ==============================================================
# FIGURA CON DISPOSICIÓN MEJORADA
# ==============================================================

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[2, 0.8], width_ratios=[2, 1, 1],
                       hspace=0.3, wspace=0.3)

# --------------------------------------------------------------
# 1. ROSA DE ONDAS (polar)
# --------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0], projection='polar')
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)
ax1.set_theta_offset(np.pi/2)

# Normalizar histograma
max_hist = np.max(hist) if np.max(hist) > 0 else 1
hist_norm = hist / max_hist

# Dibujar barras
bars = ax1.bar(bin_centers_rad, hist_norm, width=np.deg2rad(BIN_WIDTH_DEG),
               bottom=0, edgecolor='black', linewidth=0.5, alpha=0.8,
               color=plt.cm.plasma(hist_norm / np.max(hist_norm) if np.max(hist_norm)>0 else 1))

# Configurar ejes
ax1.set_ylim(0, 1.1)
ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
ax1.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
ax1.set_xlabel('Direction (degrees)', fontsize=11, labelpad=8)
ax1.set_title('Wave Direction Rose Diagram', fontsize=14, fontweight='bold', pad=15)

# Etiquetas de dirección (N, E, S, W) con mejor posicionamiento
ax1.text(np.deg2rad(0), 1.15, 'N', ha='center', va='center', fontsize=12, fontweight='bold')
ax1.text(np.deg2rad(90), 1.15, 'E', ha='center', va='center', fontsize=12, fontweight='bold')
ax1.text(np.deg2rad(180), 1.15, 'S', ha='center', va='center', fontsize=12, fontweight='bold')
ax1.text(np.deg2rad(270), 1.15, 'W', ha='center', va='center', fontsize=12, fontweight='bold')

# Colorbar para la energía (colocada fuera del gráfico polar)
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar1 = fig.colorbar(sm, ax=ax1, orientation='vertical', pad=0.1, shrink=0.7, aspect=30)
cbar1.set_label('Normalized Energy', fontsize=10)

# --------------------------------------------------------------
# 2. HISTOGRAMA DE LONGITUD DE ONDA
# --------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
wavelengths_valid = wavelengths[wavelengths > 0]
if len(wavelengths_valid) > 0:
    ax2.hist(wavelengths_valid, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Wavelength (m)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Wavelength Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    mean_wl = np.mean(wavelengths_valid)
    median_wl = np.median(wavelengths_valid)
    ax2.axvline(mean_wl, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_wl:.1f} m')
    ax2.axvline(median_wl, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_wl:.1f} m')
    ax2.legend(fontsize=8, loc='upper right')
else:
    ax2.text(0.5, 0.5, 'No valid wavelengths', ha='center', va='center', transform=ax2.transAxes)

# --------------------------------------------------------------
# 3. HISTOGRAMA DE ORIENTACIÓN (lineal)
# --------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df['orientation_deg'], bins=24, color='coral', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Orientation (deg)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Orientation Distribution', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 360)

# --------------------------------------------------------------
# 4. TABLA DE ESTADÍSTICAS (más compacta y bien ubicada)
# --------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, :])
ax4.axis('off')
# Calcular estadísticas
stats = {
    'Samples': len(df),
    'Mean Wavelength (m)': f"{np.mean(wavelengths_valid):.1f}" if len(wavelengths_valid)>0 else "N/A",
    'Std Wavelength (m)': f"{np.std(wavelengths_valid):.1f}" if len(wavelengths_valid)>0 else "N/A",
    'Mean Orientation (deg)': f"{np.mean(df['orientation_deg']):.1f}",
    'Std Orientation (deg)': f"{np.std(df['orientation_deg']):.1f}",
    'Mean Energy Ratio': f"{np.mean(df['energy_ratio']):.3f}",
    'Mean Probability': f"{np.mean(df['probability']):.3f}",
}
# Crear tabla con dos columnas
table_data = list(stats.items())
table = ax4.table(cellText=table_data, colLabels=['Parameter', 'Value'],
                  loc='center', cellLoc='center', colWidths=[0.25, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
# Color de fondo alterno
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='white', fontweight='bold')
    elif i % 2 == 0:
        cell.set_facecolor('#f0f0f0')
    else:
        cell.set_facecolor('#d9d9d9')
ax4.set_title('Wave Parameter Summary', fontsize=14, fontweight='bold', pad=15)

# --------------------------------------------------------------
# AJUSTE FINAL DEL DISEÑO Y GUARDADO
# --------------------------------------------------------------
plt.suptitle('Wave Analysis - Rose Diagram & Statistics', fontsize=16, fontweight='bold', y=1.02)
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
plt.show()

print(f"Diagrama guardado en: {OUTPUT_FIGURE}")

# ==============================================================
# RESUMEN EN CONSOLA
# ==============================================================
print("\n===== WAVE PARAMETERS SUMMARY =====")
for key, val in stats.items():
    print(f"{key}: {val}")
print("=====================================\n")