# Tanager Hyperspectral Wave Parameter Extraction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxx)

**A complete Python workflow for extracting wave parameters (wavelength, direction, energy) from Tanager hyperspectral satellite imagery, using Random Forest classification, Fourier analysis, and wavelet transforms.**

---

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características](#características)
- [Requisitos de Datos](#requisitos-de-datos)
- [Instalación](#instalación)
- [Flujo de Trabajo](#flujo-de-trabajo)
- [Uso](#uso)
- [Resultados](#resultados)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Dependencias](#dependencias)
- [Licencia](#licencia)
- [Citación](#citación)
- [Contacto](#contacto)

---

## Descripción General

Este proyecto investiga el potencial de las imágenes hiperespectrales del satélite Tanager para extraer información de olas necesaria para la **inversión batimétrica basada en ondas** en entornos costeros. El área de estudio es Santa Cruz, Galápagos, Ecuador.

El flujo de trabajo transforma las observaciones hiperespectrales de Tanager en descriptores cuantitativos de olas: **longitud de onda**, **dirección de propagación** y **energía**, que pueden usarse como entradas para relaciones de dispersión de ondas y estimar la profundidad del agua.

El pipeline integra:

- **Clasificación supervisada** (Random Forest) para aislar el dominio acuático.
- **Transformada rápida de Fourier 2D (FFT)** y **transformada wavelet** para la detección de olas.
- **Extracción de parámetros físicos** (longitud de onda, orientación, energía).
- **Caracterización de fenómenos de olas** (refracción, difracción, interferencia).

---

## Características

- **Selección interactiva de ROI** – Selecciona muestras de entrenamiento para 7 clases de cobertura del suelo (AGUA, TIERRA, NUBES, SIN_DATOS, AGUA_DULCE, SOMBRA, VEGETACIÓN) usando una interfaz gráfica.
- **Clasificación Random Forest** – Entrena y aplica un clasificador para producir un mapa de 7 clases y máscaras individuales (GeoTIFF).
- **Detección de olas** – Muestrea píxeles de agua, calcula la FFT 2D y características wavelet, y genera un mapa probabilístico de presencia de olas.
- **Extracción de parámetros** – Para píxeles con alta probabilidad de ola, extrae longitud de onda (m), orientación (grados) y relación de energía.
- **Análisis de fenómenos de olas** – Calcula gradiente local de orientación (refracción), curvatura (difracción) y varianza circular (interferencia).
- **Visualización completa** – Diagramas de rosa, mapas de superposición, mapas de probabilidad y figuras resumen.
- **Todos los resultados son compatibles con GeoTIFF** para integración en SIG.

---

## Requisitos de Datos

Necesitas los siguientes productos de Tanager (descargados desde el [Tanager STAC Browser](https://www.planet.com/data/stac/browser/tanager-core-imagery/)):

| Archivo | Descripción |
|---------|-------------|
| `*_ortho_visual.tif` | Imagen RGB ortorrectificada visual |
| `*_ortho_sr_hdf5.h5` | Cubo hiperespectral de reflectancia superficial ortorrectificado |
| `*_ortho_beta_udm.tif` | Máscara de nubes y calidad (opcional pero recomendada) |

Coloca todos los archivos en un solo directorio (por ejemplo, `D:\Arcelia\Tangent` o ajusta la variable `BASE_DIR` en los scripts).

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/tanager-wave-extraction.git
cd tanager-wave-extraction
