# ==============================================================
# EXTRA WAVE PARAMETER PLOTS
# ==============================================================
# Genera múltiples gráficos exploratorios a partir del CSV
# de parámetros de ondas.
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# ==============================================================
# CONFIGURACIÓN
# ==============================================================

CSV_PATH = r"D:\Arcelia\Tangent\wave_detection_output\wave_parameters.csv"
OUTPUT_DIR = r"D:\Arcelia\Tangent\wave_detection_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selección de gráficos a generar (True/False)
PLOTS = {
    'scatter_wavelength_energy': True,
    'scatter_wavelength_prob': True,
    'hexbin_orientation_wavelength': True,
    'hist_probability': True,
    'boxplot_orientation_by_prob': True,
    'correlation_matrix': True,
    'spatial_map_orientation': True,
    'spatial_map_wavelength': True,
    'energy_vs_probability': True,
}

# ==============================================================
# LECTURA DE DATOS
# ==============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"No se encontró el archivo: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Archivo cargado: {CSV_PATH}")
print(f"Registros totales: {len(df)}")
print(f"Columnas disponibles: {df.columns.tolist()}")

# Eliminar filas con valores atípicos (opcional)
df_clean = df[(df['wavelength_m'] > 0) & (df['wavelength_m'] < 5000)]
print(f"Registros después de limpiar longitudes de onda extremas: {len(df_clean)}")

# ==============================================================
# 1. SCATTER: Longitud de onda vs Energía
# ==============================================================
if PLOTS['scatter_wavelength_energy']:
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_clean['wavelength_m'], df_clean['energy_ratio'],
                         c=df_clean['probability'], cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Wavelength (m)', fontsize=12)
    ax.set_ylabel('Energy Ratio', fontsize=12)
    ax.set_title('Wavelength vs Energy Ratio (colored by Probability)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Probability', fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'scatter_wavelength_energy.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter guardado: {out_path}")

# ==============================================================
# 2. SCATTER: Longitud de onda vs Probabilidad
# ==============================================================
if PLOTS['scatter_wavelength_prob']:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_clean['wavelength_m'], df_clean['probability'], alpha=0.6, s=30, c='teal')
    ax.set_xlabel('Wavelength (m)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Wavelength vs Probability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'scatter_wavelength_prob.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter guardado: {out_path}")

# ==============================================================
# 3. HEXBIN: Orientación vs Longitud de onda
# ==============================================================
if PLOTS['hexbin_orientation_wavelength']:
    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(df_clean['orientation_deg'], df_clean['wavelength_m'],
                   gridsize=30, cmap='plasma', bins='log', mincnt=1)
    ax.set_xlabel('Orientation (deg)', fontsize=12)
    ax.set_ylabel('Wavelength (m)', fontsize=12)
    ax.set_title('Hexbin: Orientation vs Wavelength (log count)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Log Count', fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'hexbin_orientation_wavelength.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hexbin guardado: {out_path}")

# ==============================================================
# 4. HISTOGRAMA DE PROBABILIDAD
# ==============================================================
if PLOTS['hist_probability']:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_clean['probability'], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Wave Probability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'hist_probability.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histograma guardado: {out_path}")

# ==============================================================
# 5. BOXPLOT: Orientación agrupada por probabilidad (cuartiles)
# ==============================================================
if PLOTS['boxplot_orientation_by_prob']:
    # Dividir probabilidad en cuartiles
    df_clean['prob_quartile'] = pd.qcut(df_clean['probability'], q=4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='prob_quartile', y='orientation_deg', data=df_clean, palette='Set2', ax=ax)
    ax.set_xlabel('Probability Quartile', fontsize=12)
    ax.set_ylabel('Orientation (deg)', fontsize=12)
    ax.set_title('Orientation Distribution by Probability Quartile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'boxplot_orientation_by_prob.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot guardado: {out_path}")

# ==============================================================
# 6. MATRIZ DE CORRELACIÓN
# ==============================================================
if PLOTS['correlation_matrix']:
    cols_corr = ['freq_cycles_per_pixel', 'wavelength_m', 'orientation_deg',
                 'energy_ratio', 'probability']
    corr_df = df_clean[cols_corr].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Wave Parameters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de correlación guardada: {out_path}")

# ==============================================================
# 7. MAPA ESPACIAL: Orientación (solo puntos muestreados)
# ==============================================================
if PLOTS['spatial_map_orientation']:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df_clean['col'], df_clean['row'], c=df_clean['orientation_deg'],
                         cmap='twilight', s=10, alpha=0.7, vmin=0, vmax=360)
    ax.set_xlabel('Column (pixel)', fontsize=12)
    ax.set_ylabel('Row (pixel)', fontsize=12)
    ax.set_title('Spatial Distribution of Wave Orientation', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Orientation (deg)', fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'spatial_map_orientation.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa espacial de orientación guardado: {out_path}")

# ==============================================================
# 8. MAPA ESPACIAL: Longitud de onda
# ==============================================================
if PLOTS['spatial_map_wavelength']:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df_clean['col'], df_clean['row'], c=df_clean['wavelength_m'],
                         cmap='viridis', s=10, alpha=0.7)
    ax.set_xlabel('Column (pixel)', fontsize=12)
    ax.set_ylabel('Row (pixel)', fontsize=12)
    ax.set_title('Spatial Distribution of Wavelength', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Wavelength (m)', fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'spatial_map_wavelength.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa espacial de longitud de onda guardado: {out_path}")

# ==============================================================
# 9. ENERGÍA vs PROBABILIDAD (con densidad)
# ==============================================================
if PLOTS['energy_vs_probability']:
    fig, ax = plt.subplots(figsize=(10, 6))
    xy = np.vstack([df_clean['energy_ratio'], df_clean['probability']])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = df_clean['energy_ratio'][idx], df_clean['probability'][idx], z[idx]
    scatter = ax.scatter(x, y, c=z, s=30, cmap='magma', alpha=0.7)
    ax.set_xlabel('Energy Ratio', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Energy Ratio vs Probability (colored by density)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density', fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'energy_vs_probability.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico energía vs probabilidad guardado: {out_path}")

print("\nTodos los gráficos seleccionados han sido generados.")
print(f"Ubicación: {OUTPUT_DIR}")