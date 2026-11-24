from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
import streamlit as st
import torch
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from copernicus_flood.dataset.manifest import load_training_manifest
from copernicus_flood.inference.predict import predict_raster
from copernicus_flood.ingestion.registry import IngestionError
from copernicus_flood.preprocessing.labels import read_product_layer


DATASETS = ROOT / "data" / "processed" / "EMSR865"
MODELS = ROOT / "data" / "models" / "EMSR865"
PRODUCTS = ROOT / "data" / "raw" / "cems" / "EMSR865" / "AOI01"
RESULTS = ROOT / "data" / "inference" / "EMSR865" / "demo"


def create_patch_stack(source_path: Path, record: dict, output_path: Path) -> np.ndarray:
    window = Window(
        int(record["col_off"]),
        int(record["row_off"]),
        int(record["width"]),
        int(record["height"]),
    )
    with rasterio.open(source_path) as source:
        if (
            window.col_off < 0
            or window.row_off < 0
            or window.width <= 0
            or window.height <= 0
            or window.col_off + window.width > source.width
            or window.row_off + window.height > source.height
        ):
            raise IngestionError("La patch selezionata è fuori dallo stack.")
        values = source.read((1, 2), window=window)
        mask = source.dataset_mask(window=window)
        profile = source.profile.copy()
        profile.update(
            width=int(window.width),
            height=int(window.height),
            transform=source.window_transform(window),
            tiled=False,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        with rasterio.open(output_path, "w", **profile) as output:
            output.write(values)
            output.write_mask(mask)
            for band, description in enumerate(source.descriptions, start=1):
                if description:
                    output.set_band_description(band, description)

    vv = values[0].astype(np.float32)
    valid = (mask > 0) & np.isfinite(vv)
    preview = np.zeros(vv.shape, dtype=np.uint8)
    if np.any(valid):
        transformed = np.log1p(np.maximum(vv[valid], 0))
        low, high = np.percentile(transformed, (2, 98))
        if high > low:
            preview[valid] = (
                np.clip((transformed - low) / (high - low), 0, 1) * 255
            ).astype(np.uint8)
    return preview


def run_demo(
    dataset_dir: Path,
    dataset_info: dict,
    record: dict,
    checkpoint: Path,
    threshold: float,
) -> dict:
    association = dataset_info.get("scene_association", {})
    product_id = association.get("product_id")
    if not isinstance(product_id, int):
        raise IngestionError("ID del prodotto CEMS assente nel manifesto.")
    archives = list(PRODUCTS.glob(f"*_id{product_id}.zip"))
    if len(archives) != 1:
        raise IngestionError(f"Archivio CEMS unico non trovato per il prodotto {product_id}.")
    source_path = Path(dataset_info["input_stack"])
    if not source_path.is_file():
        raise IngestionError(f"Stack Sentinel-1 non trovato: {source_path}")

    output_dir = RESULTS / uuid4().hex
    output_dir.mkdir(parents=True)
    stack_path = output_dir / "input_patch.tif"
    preview = create_patch_stack(source_path, record, stack_path)
    aoi_path = output_dir / "aoi.geojson"
    read_product_layer(archives[0], "areaOfInterestA").to_file(
        aoi_path, driver="GeoJSON", index=False
    )
    result = predict_raster(
        argparse.Namespace(
            input_stack=stack_path,
            aoi_vector=aoi_path,
            checkpoint=checkpoint,
            output_dir=output_dir,
            tile_size=256,
            threshold=threshold,
            minimum_polygon_pixels=4,
            preview_max_dimension=512,
            device="cpu",
        )
    )
    context = {
        "dataset": str(dataset_dir),
        "source_stack": str(source_path),
        "patch_id": record["patch_id"],
        "split": record["split"],
        "row_off": record["row_off"],
        "col_off": record["col_off"],
        "height": record["height"],
        "width": record["width"],
        "prediction": result,
    }
    (output_dir / "demo_manifest.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"preview": preview, "context": context, "output_dir": output_dir}


st.set_page_config(page_title="Flood Lab | EMSR865", page_icon="🌊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    .eyebrow {color: #168c9b; font-size: .78rem; font-weight: 700;
              letter-spacing: .16em; text-transform: uppercase;}
    .hero-title {font-size: 2.8rem; font-weight: 750; line-height: 1.1;
                 margin: .35rem 0 .6rem;}
    .hero-copy {opacity: .8; font-size: 1.05rem; max-width: 720px;}
    div[data-testid="stMetric"] {background: #f3f8fa; border: 1px solid #dcebed;
       padding: 1rem; border-radius: .75rem;}
    div[data-testid="stMetric"] p {color: #183541;}
    </style>
    <div class="eyebrow">Copernicus EMSR865 · Sentinel-1</div>
    <div class="hero-title">Flood Lab</div>
    <div class="hero-copy">Esplora una previsione di allagamento su una patch
    VV/VH verificata. Una demo locale, dal dato satellitare alla mappa.</div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Demo esplorativa, non operativa. Il background negativo è un'assunzione "
    "didattica e lo split è interno alla sola attivazione EMSR865."
)

datasets = sorted(path.parent for path in DATASETS.glob("*/dataset_manifest.json"))
checkpoints = sorted(MODELS.glob("**/checkpoint_best.pt"))
if not datasets:
    st.error("Nessun dataset disponibile. Eseguire prima il preprocessing EMSR865.")
    st.stop()
if not checkpoints:
    st.error("Nessun checkpoint_best.pt disponibile. Eseguire prima il training.")
    st.stop()

with st.sidebar:
    st.markdown("### Configura la demo")
    dataset_dir = st.selectbox("Dataset", datasets, format_func=lambda path: path.name)
    checkpoint = st.selectbox(
        "Modello",
        checkpoints,
        format_func=lambda path: str(path.relative_to(MODELS)),
    )
    threshold = st.slider("Soglia di allagamento", 0.05, 0.95, 0.50, 0.05)
    st.caption("Le elaborazioni usano la CPU e restano sul computer locale.")

try:
    dataset_info, records, dataset_sha256 = load_training_manifest(dataset_dir)
    checkpoint_info = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_info, dict):
        raise IngestionError("Checkpoint non valido.")
except (IngestionError, OSError, ValueError, RuntimeError) as error:
    st.error(f"Impossibile caricare dataset o modello: {error}")
    st.stop()

if checkpoint_info.get("dataset_manifest_sha256") != dataset_sha256:
    st.warning(
        "Il checkpoint è stato addestrato su un manifesto diverso da quello "
        "selezionato. La previsione non costituisce una valutazione del modello."
    )
if "acceptance-smoke" in checkpoint.parts:
    st.info("Checkpoint di smoke test: addestrato per una sola epoca, non operativo.")

split = st.selectbox("Split spaziale", ("test", "validation", "train"))
patches = [record for record in records if record["split"] == split]
patch_by_id = {record["patch_id"]: record for record in patches}
patch_id = st.selectbox(
    "Patch da analizzare",
    list(patch_by_id),
    help="La patch è una finestra del raster già preparato; la maschera osservata non è usata come input del modello.",
)
record = patch_by_id[patch_id]
st.caption(
    f"Finestra {record['width']} × {record['height']} pixel · "
    f"riga {record['row_off']} · colonna {record['col_off']}"
)

selection = (str(dataset_dir), str(checkpoint), patch_id, float(threshold))
if st.button("Avvia inferenza", type="primary"):
    try:
        with st.spinner("Elaborazione della patch in corso…"):
            result = run_demo(dataset_dir, dataset_info, record, checkpoint, threshold)
        st.session_state["demo_result"] = result
        st.session_state["demo_selection"] = selection
    except (IngestionError, OSError, ValueError, RuntimeError, KeyError) as error:
        st.session_state.pop("demo_result", None)
        st.error(f"Inferenza non riuscita: {error}")

if "demo_result" in st.session_state and st.session_state.get("demo_selection") == selection:
    result = st.session_state["demo_result"]
    prediction = result["context"]["prediction"]
    output_dir = Path(result["output_dir"])
    st.markdown("### Risultato della patch")
    columns = st.columns(3)
    columns[0].metric("Pixel validi", f"{prediction['valid_pixel_count']:,}".replace(",", "."))
    columns[1].metric("Pixel previsti allagati", f"{prediction['flood_pixel_count']:,}".replace(",", "."))
    columns[2].metric("Poligoni", prediction["flood_polygon_count"])
    left, right = st.columns(2)
    with left:
        st.image(result["preview"], caption="Sentinel-1 VV · ampiezza logaritmica")
    with right:
        st.image(str(output_dir / "flood_preview.png"), caption="Predizione · blu: allagamento, grigio: background, trasparente: escluso")
    st.caption("L'anteprima copre solo la patch selezionata, non l'intera AOI.")
    st.markdown("#### Scarica i risultati")
    files = [
        ("Maschera GeoTIFF", "flood_mask.tif", "image/tiff"),
        ("Probabilità GeoTIFF", "flood_probability.tif", "image/tiff"),
        ("Poligoni GeoJSON", "flood_polygons.geojson", "application/geo+json"),
        ("Manifesto demo", "demo_manifest.json", "application/json"),
    ]
    if prediction["geopackage"]:
        files.append(("Poligoni GeoPackage", "flood_polygons.gpkg", "application/geopackage+sqlite3"))
    for label, name, mime in files:
        with (output_dir / name).open("rb") as file:
            st.download_button(label, file.read(), file_name=name, mime=mime)
    st.caption(f"Risultati salvati in: {output_dir}")
else:
    st.info("Seleziona una patch e avvia l'inferenza per visualizzare la mappa.")
