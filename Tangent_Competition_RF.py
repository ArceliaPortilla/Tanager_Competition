# ==============================================================
# RANDOM FOREST CLASSIFICATION OF TANAGER HYPERSPECTRAL IMAGE
# Interactive ROI selection
#
# Classes:
#   0 = WATER
#   1 = LAND
#   2 = CLOUD
#   3 = NO_DATA
#   4 = FRESH_WATER
#   5 = SHADOW
#   6 = VEGETATION
#
# VERSION: with loop until at least one ROI is selected
# ==============================================================

import os
import glob
import warnings

import numpy as np
import rasterio
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURACION DE CARPETA Y ARCHIVOS
# ==============================================================

BASE_DIR = r"D:\Arcelia\Tangent"
OUTPUT_DIR = os.path.join(BASE_DIR, "wave_detection_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Buscar archivo de reflectancia (priorizar ortho sobre basic)
sr_patterns = [
    os.path.join(BASE_DIR, "*ortho_sr_hdf5.h5"),
    os.path.join(BASE_DIR, "*ortho_sr_hdf5.hdf"),
    os.path.join(BASE_DIR, "*basic_sr_hdf5.h5"),
    os.path.join(BASE_DIR, "*basic_sr_hdf5.hdf"),
]
sr_files = []
for p in sr_patterns:
    sr_files.extend(glob.glob(p))

if not sr_files:
    raise FileNotFoundError(
        "No se encontro ningun archivo *_sr_hdf5.h5 en " + BASE_DIR
    )
SR_HDF5 = sr_files[0]
print("Archivo de reflectancia seleccionado:", os.path.basename(SR_HDF5))

# Buscar archivo de mascara de nubes (priorizar ortho sobre basic)
udm_patterns = [
    os.path.join(BASE_DIR, "*ortho_beta_udm.tif"),
    os.path.join(BASE_DIR, "*ortho_beta_udm.tiff"),
    os.path.join(BASE_DIR, "*basic_beta_udm.tif"),
    os.path.join(BASE_DIR, "*basic_beta_udm.tiff"),
]
udm_files = []
for p in udm_patterns:
    udm_files.extend(glob.glob(p))

if udm_files:
    UDM_TIFF = udm_files[0]
    print("Archivo de mascara de nubes seleccionado:", os.path.basename(UDM_TIFF))
else:
    UDM_TIFF = None
    print("ADVERTENCIA: No se encontro *_beta_udm.tif. Se usara mascara predeterminada.")

# Buscar archivo visual RGB (ortho_visual.tif)
visual_patterns = [
    os.path.join(BASE_DIR, "*ortho_visual.tif"),
    os.path.join(BASE_DIR, "*ortho_visual.tiff"),
]
visual_files = []
for p in visual_patterns:
    visual_files.extend(glob.glob(p))

if visual_files:
    VISUAL_TIFF = visual_files[0]
    print("Archivo visual RGB seleccionado:", os.path.basename(VISUAL_TIFF))
else:
    VISUAL_TIFF = None
    print("ADVERTENCIA: No se encontro ortho_visual.tif. Se usara RGB construido.")

# ==============================================================
# CONFIGURACION ESPECTRAL Y DE CLASIFICACION (7 clases)
# ==============================================================

TARGET_WL = {
    "blue": 490,
    "green": 560,
    "red": 665,
    "nir": 865,
    "swir1": 1610,
}

FEATURE_BANDS = ["blue", "green", "red", "nir", "swir1"]

# Clases actualizadas (7 clases)
CLASS_NAMES = {
    0: "WATER",
    1: "LAND",
    2: "CLOUD",
    3: "NO_DATA",
    4: "FRESH_WATER",
    5: "SHADOW",
    6: "VEGETATION"
}

CLASS_COLORS = {
    0: "blue",
    1: "saddlebrown",
    2: "gray",
    3: "black",
    4: "cyan",
    5: "violet",
    6: "green"
}

# ==============================================================
# CARGA DE DATOS CON H5PY + RASTERIO
# ==============================================================

def load_tanager_data_h5py(hdf5_path, target_wl, cloud_tiff_path=None):
    """
    Carga reflectancia desde HDF5 usando h5py.
    Si no encuentra longitudes de onda, usa rango lineal.
    """
    with h5py.File(hdf5_path, 'r') as f:
        # ---- Buscar dataset de reflectancia/radiancia ----
        ds_name = None
        def find_sr(name, obj):
            if isinstance(obj, h5py.Dataset):
                if 'surface_reflectance' in name or 'radiance' in name:
                    return name
            return None
        ds_name = f.visititems(find_sr)

        if ds_name is None:
            raise ValueError("No se encontro 'surface_reflectance' ni 'radiance' en el HDF5.")

        print("Dataset de reflectancia encontrado:", ds_name)
        dataset = f[ds_name]
        cube = dataset[:].astype(np.float32)
        height, width = cube.shape[1], cube.shape[2]
        num_bands = cube.shape[0]

        # ---- Buscar longitudes de onda ----
        wavelengths = None

        # Intento 1: Misma ruta
        base_path = os.path.dirname(ds_name)
        for candidate in ['wavelengths', 'wavelength', 'lambda']:
            test_path = base_path + '/' + candidate
            if test_path in f:
                wavelengths = f[test_path][:]
                print("Longitudes de onda encontradas en:", test_path)
                break

        # Intento 2: Buscar recursivo
        if wavelengths is None:
            def find_wl(name, obj):
                if isinstance(obj, h5py.Dataset) and 'wavelength' in name.lower():
                    return name
                return None
            wl_name = f.visititems(find_wl)
            if wl_name is not None:
                wavelengths = f[wl_name][:]
                print("Longitudes de onda encontradas en:", wl_name)

        # Intento 3: Atributos del dataset
        if wavelengths is None:
            if 'wavelengths' in dataset.attrs:
                wavelengths = dataset.attrs['wavelengths']
                print("Longitudes de onda leidas desde atributos del dataset.")
            elif 'wavelength' in dataset.attrs:
                wavelengths = dataset.attrs['wavelength']
                print("Longitudes de onda leidas desde atributo 'wavelength'.")

        # Si no se encontraron, crear rango lineal (400-2500 nm)
        if wavelengths is None:
            wavelengths = np.linspace(400, 2500, num_bands)
            print("ADVERTENCIA: Usando rango lineal de longitudes de onda (400-2500 nm).")

        wavelengths = np.array(wavelengths)
        if wavelengths.ndim > 1:
            wavelengths = wavelengths.flatten()
        print("Numero de bandas:", len(wavelengths))

        # ---- Extraer bandas por longitud de onda ----
        bands = {}
        print("\nBandas seleccionadas:")
        for name, target in target_wl.items():
            idx = np.argmin(np.abs(wavelengths - target))
            bands[name] = cube[idx, :, :]
            print(
                "  {:6s} -> target = {:4d} nm | indice = {:3d} | lambda = {:.2f} nm".format(
                    name, target, idx, wavelengths[idx]
                )
            )

        # ---- Transformacion y CRS ----
        transform = None
        crs = None
        if 'transform' in f.attrs:
            transform = f.attrs['transform']
        elif 'geotransform' in f.attrs:
            transform = f.attrs['geotransform']
        else:
            from rasterio.transform import from_origin
            transform = from_origin(0, 0, 1, 1)
            print("ADVERTENCIA: No se encontro transformacion, se usara dummy.")

        if 'crs' in f.attrs:
            crs = f.attrs['crs']
        elif 'projection' in f.attrs:
            crs = f.attrs['projection']
        else:
            crs = None
            print("ADVERTENCIA: No se encontro CRS, se usara None.")

    # ---- Leer mascara de nubes desde COG (si existe) ----
    cloud_mask = None
    if cloud_tiff_path is not None and os.path.exists(cloud_tiff_path):
        with rasterio.open(cloud_tiff_path) as src:
            cloud_data = src.read(1).astype(np.uint8)
            cloud_mask = (cloud_data == 0)
            print("Mascara de nubes cargada desde COG. Dimensiones:", cloud_mask.shape)
    else:
        cloud_mask = np.ones((height, width), dtype=bool)
        print("ADVERTENCIA: No se uso mascara de nubes externa. Se asume todo despejado.")

    if cloud_mask.shape != (height, width):
        raise ValueError(
            "Dimensiones de la mascara {} no coinciden con la imagen ({}, {})".format(
                cloud_mask.shape, height, width
            )
        )

    return bands, transform, crs, height, width, cloud_mask


# ==============================================================
# NORMALIZACION PARA VISUALIZACION
# ==============================================================

def normalize_percentile(img, pmin=2, pmax=98):
    valid = np.isfinite(img)
    if not np.any(valid):
        return np.zeros_like(img, dtype=np.float32)
    vmin, vmax = np.nanpercentile(img[valid], [pmin, pmax])
    if vmax == vmin:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - vmin) / (vmax - vmin)
    return np.clip(out, 0, 1)


# ==============================================================
# CARGA DE IMAGEN VISUAL RGB (ortho_visual.tif)
# ==============================================================

def load_visual_rgb(visual_path):
    """Carga el ortho_visual.tif y lo normaliza para mostrar."""
    with rasterio.open(visual_path) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
        r_norm = normalize_percentile(r)
        g_norm = normalize_percentile(g)
        b_norm = normalize_percentile(b)
        rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
        return rgb


# ==============================================================
# SELECTOR INTERACTIVO DE ROIS (con 7 clases)
# ==============================================================

class ROISelector:
    def __init__(self, img_display, band_data, band_names):
        self.img_display = img_display
        self.band_data = band_data
        self.band_names = band_names
        self.height, self.width = img_display.shape[:2]

        self.current_class = None
        self.rois = []
        self.rectangles = []
        self.finished = False
        self.cancelled = False

        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.ax.imshow(img_display, origin="upper")
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(self.height, 0)
        self.ax.set_xlabel("Column (pixel)")
        self.ax.set_ylabel("Row (pixel)")
        self.update_title()

        self.rectangle_selector = RectangleSelector(
            self.ax, self.onselect,
            useblit=True, button=[1],
            minspanx=3, minspany=3,
            spancoords="pixels", interactive=True
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def update_title(self):
        if self.current_class is None:
            title = (
                "PRIMERO SELECCIONE CLASE: [W] WATER | [L] LAND | [C] CLOUD | "
                "[N] NO_DATA | [F] FRESH_WATER | [S] SHADOW | [V] VEGETATION | "
                "[ENTER] FINISH | [ESC] CANCEL"
            )
        else:
            title = (
                "CURRENT CLASS: {} | [W] WATER | [L] LAND | [C] CLOUD | "
                "[N] NO_DATA | [F] FRESH_WATER | [S] SHADOW | [V] VEGETATION | "
                "[D] UNDO | [ENTER] FINISH | [ESC] CANCEL".format(
                    CLASS_NAMES[self.current_class]
                )
            )
        self.ax.set_title(title, fontsize=11)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key is None:
            return
        key = event.key.lower()

        if key == "w":
            self.current_class = 0
            print("\n>>> Current class: WATER")
            self.update_title()
        elif key == "l":
            self.current_class = 1
            print("\n>>> Current class: LAND")
            self.update_title()
        elif key == "c":
            self.current_class = 2
            print("\n>>> Current class: CLOUD")
            self.update_title()
        elif key == "n":
            self.current_class = 3
            print("\n>>> Current class: NO_DATA")
            self.update_title()
        elif key == "f":
            self.current_class = 4
            print("\n>>> Current class: FRESH_WATER")
            self.update_title()
        elif key == "s":
            self.current_class = 5
            print("\n>>> Current class: SHADOW")
            self.update_title()
        elif key == "v":   # NUEVA TECLA PARA VEGETATION
            self.current_class = 6
            print("\n>>> Current class: VEGETATION")
            self.update_title()
        elif key == "d":
            if len(self.rois) == 0:
                print("\nNo ROI available to remove.")
                return
            roi = self.rois.pop()
            rect = self.rectangles.pop()
            rect.remove()
            self.fig.canvas.draw_idle()
            print("\nRemoved ROI: {}".format(CLASS_NAMES[roi['class_id']]))
        elif key == "enter":
            self.finished = True
            self.rectangle_selector.set_active(False)
            plt.close(self.fig)
        elif key == "escape":
            self.cancelled = True
            self.rois = []
            self.rectangle_selector.set_active(False)
            plt.close(self.fig)

    def onselect(self, eclick, erelease):
        if self.current_class is None:
            print("\nWARNING: Primero seleccione una clase con W, L, C, N, F, S o V.")
            return
        if (eclick.xdata is None or eclick.ydata is None or
                erelease.xdata is None or erelease.ydata is None):
            return

        x1 = int(np.floor(eclick.xdata))
        x2 = int(np.ceil(erelease.xdata))
        y1 = int(np.floor(eclick.ydata))
        y2 = int(np.ceil(erelease.ydata))

        xmin = max(0, min(x1, x2))
        xmax = min(self.width, max(x1, x2))
        ymin = max(0, min(y1, y2))
        ymax = min(self.height, max(y1, y2))

        if xmax <= xmin or ymax <= ymin:
            print("\nWARNING: Rectangulo invalido (demasiado pequeno).")
            return

        roi = {
            "class_id": self.current_class,
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax
        }
        self.rois.append(roi)

        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor=CLASS_COLORS[self.current_class], linewidth=2
        )
        self.ax.add_patch(rect)
        self.rectangles.append(rect)
        self.fig.canvas.draw_idle()

        pixels = (xmax - xmin) * (ymax - ymin)
        print(
            "\nROI added:\n  Class   : {}\n  Rows    : {}:{}\n  Columns : {}:{}\n  Pixels  : {}".format(
                CLASS_NAMES[self.current_class], ymin, ymax, xmin, xmax, pixels
            )
        )

    def get_samples(self):
        if self.cancelled:
            return None, None
        if len(self.rois) == 0:
            return None, None

        class_map = np.full((self.height, self.width), -1, dtype=np.int16)
        overlap_count = 0

        for roi in self.rois:
            class_id = roi["class_id"]
            xmin, xmax = roi["xmin"], roi["xmax"]
            ymin, ymax = roi["ymin"], roi["ymax"]
            current = class_map[ymin:ymax, xmin:xmax]
            overlap = (current != -1)
            overlap_count += np.sum(overlap)
            current[~overlap] = class_id
            class_map[ymin:ymax, xmin:xmax] = current

        if overlap_count > 0:
            print(
                "\nWARNING: {} pixels belonged to overlapping ROIs. "
                "Those pixels were assigned to the FIRST ROI that selected them.".format(
                    overlap_count
                )
            )

        rows, cols = np.where(class_map >= 0)
        labels = class_map[rows, cols]

        X = np.column_stack([
            self.band_data[name][rows, cols] for name in self.band_names
        ])

        valid = np.isfinite(X).all(axis=1)
        X = X[valid]
        labels = labels[valid]

        selected_coords = np.column_stack([rows[valid], cols[valid]])
        _, unique_idx = np.unique(selected_coords, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        X = X[unique_idx]
        labels = labels[unique_idx]

        print("\nTraining samples:")
        for class_id in sorted(np.unique(labels)):
            n = np.sum(labels == class_id)
            print("  {:6s}: {:,} pixels".format(CLASS_NAMES[class_id], n))

        return X, labels


# ==============================================================
# FUNCION PARA REPETIR LA SELECCION HASTA TENER MUESTRAS
# ==============================================================

def select_rois_loop(rgb, bands):
    while True:
        selector = ROISelector(rgb, bands, FEATURE_BANDS)
        X, y = selector.get_samples()
        if X is not None and len(X) > 0:
            return X, y
        else:
            print("\n" + "=" * 60)
            print("No se seleccionaron ROIs. Vuelva a intentarlo.")
            print("Recuerde: primero pulse W, L, C, N, F, S o V para elegir clase,")
            print("luego arrastre el raton para dibujar rectangulos.")
            print("Presione ENTER cuando haya terminado.")
            print("=" * 60)
            resp = input("Presione ENTER para reintentar o 'q' para salir: ")
            if resp.lower() == 'q':
                raise RuntimeError("Seleccion cancelada por el usuario.")


# ==============================================================
# EJECUCION PRINCIPAL
# ==============================================================

print("\n" + "=" * 60)
print("INICIANDO PROCESO DE CLASIFICACION TANAGER (7 CLASES)")
print("=" * 60)

# Cargar datos espectrales
bands, transform, crs, height, width, cloud_mask = load_tanager_data_h5py(
    SR_HDF5, TARGET_WL, cloud_tiff_path=UDM_TIFF
)

print("\nDimensiones de la imagen: {} x {}".format(height, width))
print("CRS:", crs)

# Verificar bandas
for name in FEATURE_BANDS:
    if bands[name].shape != (height, width):
        raise ValueError(
            "Band {} tiene dimensiones {}, se esperaba ({}, {})".format(
                name, bands[name].shape, height, width
            )
        )

# ---- Cargar imagen RGB para visualizacion ----
if VISUAL_TIFF is not None and os.path.exists(VISUAL_TIFF):
    rgb = load_visual_rgb(VISUAL_TIFF)
    print("Visualizacion RGB cargada desde ortho_visual.tif")
else:
    rgb = np.stack([
        normalize_percentile(bands["red"]),
        normalize_percentile(bands["green"]),
        normalize_percentile(bands["blue"])
    ], axis=-1)
    print("Visualizacion RGB construida desde bandas espectrales.")

# ---- Seleccion interactiva de muestras (con bucle) ----
print("\n" + "=" * 60)
print("SELECCION INTERACTIVA DE MUESTRAS")
print("=" * 60)
print("\nInstrucciones:")
print("  W      -> WATER (agua salada)")
print("  L      -> LAND (tierra)")
print("  C      -> CLOUD (nubes)")
print("  N      -> NO_DATA (sin datos)")
print("  F      -> FRESH_WATER (agua dulce)")
print("  S      -> SHADOW (sombras)")
print("  V      -> VEGETATION")
print("  D      -> DESHACER ULTIMO ROI")
print("  ENTER  -> TERMINAR")
print("  ESC    -> CANCELAR")
print("\nIMPORTANTE: Los rectangulos NO deben superponerse.")
print("   Cada pixel solo puede pertenecer a una clase.")
print("   Primero seleccione una clase con su tecla, luego dibuje.")
print("=" * 60)

X, y = select_rois_loop(rgb, bands)

print("\nTotal de pixeles de entrenamiento: {:,}".format(len(X)))
print("Caracteristicas espectrales: {}".format(X.shape[1]))

# ---- Distribucion de clases ----
unique, counts = np.unique(y, return_counts=True)
print("\nDistribucion de clases:")
for cls, count in zip(unique, counts):
    print("  {:6s}: {:,} pixeles".format(CLASS_NAMES[cls], count))

required_classes = set(range(7))  # Ahora 7 clases (0 a 6)
available_classes = set(np.unique(y))
missing_classes = required_classes - available_classes
if missing_classes:
    print("\nADVERTENCIA: Faltan muestras para las clases:", 
          ", ".join(CLASS_NAMES[c] for c in sorted(missing_classes)))
    print("El clasificador se entrenara solo con las clases disponibles.")

# ---- Division entrenamiento/validacion ----
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nEntrenamiento: {:,} pixeles".format(len(X_train)))
print("Validacion:   {:,} pixeles".format(len(X_val)))

# ---- Random Forest ----
print("\n" + "=" * 60)
print("ENTRENANDO RANDOM FOREST...")
print("=" * 60)

clf = RandomForestClassifier(
    n_estimators=300,
    criterion="gini",
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
clf.fit(X_train, y_train)
print("Random Forest entrenado.")

# ---- Validacion ----
y_val_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
cm = confusion_matrix(y_val, y_val_pred, labels=list(range(7)))  # 7 clases

print("\n" + "=" * 60)
print("RESULTADOS DE VALIDACION")
print("=" * 60)
print("\nExactitud global: {:.4f}".format(accuracy))
print("\nMatriz de confusion (filas=real, columnas=predicho):")
print(cm)
print("\nReporte de clasificacion:")
target_names = [CLASS_NAMES[i] for i in range(7) if i in available_classes]
print(classification_report(
    y_val, y_val_pred,
    labels=list(available_classes),
    target_names=target_names,
    digits=4,
    zero_division=0
))

# ---- Importancia de bandas ----
print("\nImportancia de bandas:")
for name, importance in zip(FEATURE_BANDS, clf.feature_importances_):
    print("  {:6s}: {:.4f}".format(name, importance))

# ---- Clasificacion de imagen completa ----
print("\n" + "=" * 60)
print("CLASIFICANDO IMAGEN COMPLETA...")
print("=" * 60)

X_full = np.column_stack([bands[name].ravel() for name in FEATURE_BANDS])
valid = np.isfinite(X_full).all(axis=1)
print("Pixeles validos: {:,}".format(np.sum(valid)))
print("Pixeles invalidos (NaN): {:,}".format(np.sum(~valid)))

y_pred = np.full(height * width, -1, dtype=np.int16)
y_pred[valid] = clf.predict(X_full[valid])
class_map = y_pred.reshape(height, width)

# ---- Guardar mapa de clasificacion ----
print("\nGuardando resultados...")
profile = {
    "driver": "GTiff",
    "dtype": "int16",
    "width": width,
    "height": height,
    "count": 1,
    "crs": crs,
    "transform": transform,
    "compress": "lzw",
    "nodata": -1
}

classification_path = os.path.join(OUTPUT_DIR, "random_forest_classification.tif")
with rasterio.open(classification_path, "w", **profile) as dst:
    dst.write(class_map, 1)
print("  Clasificacion:", classification_path)

# ---- Guardar mascaras individuales ----
mask_profile = profile.copy()
mask_profile["dtype"] = "uint8"
mask_profile["nodata"] = 0

for class_id, class_name in CLASS_NAMES.items():
    mask = (class_map == class_id).astype(np.uint8) * 255
    out_path = os.path.join(OUTPUT_DIR, "mask_{}.tif".format(class_name.lower()))
    with rasterio.open(out_path, "w", **mask_profile) as dst:
        dst.write(mask, 1)
    print("  Mascara {}: {}".format(class_name, out_path))

# ---- Guardar muestras de entrenamiento ----
training_data_path = os.path.join(OUTPUT_DIR, "training_samples.npz")
np.savez_compressed(
    training_data_path,
    X=X, y=y,
    feature_names=np.array(FEATURE_BANDS)
)
print("  Muestras:", training_data_path)

# ---- Visualizacion final (3 paneles: RGB, Water Mask, Classification) ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 fila, 3 columnas
ax1, ax2, ax3 = axes

# 1. Imagen original RGB
ax1.imshow(rgb, origin="upper")
ax1.set_title("Original RGB Image", fontsize=14, fontweight='bold')
ax1.axis("off")

# 2. Máscara de agua (solo clase 0)
water_mask_display = (class_map == 0)
ax2.imshow(water_mask_display, cmap="Blues", origin="upper", vmin=0, vmax=1)
ax2.set_title("Water Mask (Class 0)", fontsize=14, fontweight='bold')
ax2.axis("off")

# 3. Clasificación completa con leyenda en inglés (7 clases)
# Colormap para 7 clases
cmap = ListedColormap([
    "blue",        # WATER
    "saddlebrown", # LAND
    "gray",        # CLOUD
    "black",       # NO_DATA
    "cyan",        # FRESH_WATER
    "violet",      # SHADOW
    "green"        # VEGETATION
])
masked_class_map = np.ma.masked_where(class_map < 0, class_map)
im = ax3.imshow(masked_class_map, cmap=cmap, vmin=0, vmax=6, origin="upper")
ax3.set_title("Random Forest Classification (7 classes)", fontsize=14, fontweight='bold')
ax3.axis("off")

# Añadir leyenda con los nombres de las clases en inglés
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cmap(i), label=CLASS_NAMES[i]) for i in range(7)]
ax3.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
           
           title='Classes', fontsize=10, title_fontsize=12)

plt.tight_layout()
figure_path = os.path.join(OUTPUT_DIR, "classification_results.png")
plt.savefig(figure_path, dpi=200, bbox_inches='tight')
print("  Figura:", figure_path)
plt.show()

# ---- Resumen final ----
print("\n" + "=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print("\nExactitud global en validacion: {:.4f}".format(accuracy))
print("\nTodos los resultados estan en:", os.path.abspath(OUTPUT_DIR))
print("=" * 60)