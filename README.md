# Tanager Hyperspectral Wave Parameter Extraction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.22151075))

**A complete Python workflow for extracting wave parameters (wavelength, direction, energy) from Tanager hyperspectral satellite imagery, using Random Forest classification, Fourier analysis, and wavelet transforms.**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Workflow](#workflow)
- [Usage](#usage)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

This project investigates the potential of hyperspectral satellite imagery from Tanager to extract wave information needed for **wave‑based bathymetric inversion** in coastal environments. The study area is Santa Cruz, Galápagos, Ecuador.

The workflow transforms Tanager hyperspectral observations into quantitative wave descriptors—**wavelength**, **propagation direction**, and **energy**—which can subsequently be used as inputs to wave‑dispersion relationships to estimate water depth.

The pipeline integrates:

- **Supervised classification** (Random Forest) for water‑domain isolation.
- **2D Fast Fourier Transform (FFT)** and **Wavelet transform** for wave detection.
- **Physical parameter extraction** (wavelength, orientation, energy).
- **Wave‑phenomenon characterisation** (refraction, diffraction, interference).

---

## Features

- **Interactive ROI Selection** – Select training samples for 7 land‑cover classes (WATER, LAND, CLOUD, NO_DATA, FRESH_WATER, SHADOW, VEGETATION) using a graphical interface.
- **Random Forest Classification** – Train and apply a classifier to produce a 7‑class map and individual masks (GeoTIFF).
- **Wave Detection** – Sample water pixels, compute 2D FFT and wavelet features, and generate a probabilistic wave‑presence map.
- **Parameter Extraction** – For high‑probability wave pixels, extract wavelength (m), orientation (degrees), and energy ratio.
- **Wave Phenomena Analysis** – Compute local gradient of orientation (refraction), curvature (diffraction), and circular variance (interference).
- **Comprehensive Visualisation** – Rose diagrams, overlay plots, probability maps, and summary figures.
- **All outputs are GeoTIFF‑ready** for GIS integration.

---

## Data Requirements

You need the following Tanager products (downloaded from the [Tanager STAC Browser](https://www.planet.com/data/stac/browser/tanager-core-imagery/)):

| File | Description |
|------|-------------|
| `*_ortho_visual.tif` | Orthorectified visual RGB image |
| `*_ortho_sr_hdf5.h5` | Orthorectified surface‑reflectance hyperspectral cube |
| `*_ortho_beta_udm.tif` | Cloud and quality mask (optional but recommended) |

Place all files in a single directory (e.g., `D:\Arcelia\Tangent` or adjust the `BASE_DIR` variable in the scripts).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your_username/tanager-wave-extraction.git
cd tanager-wave-extraction

