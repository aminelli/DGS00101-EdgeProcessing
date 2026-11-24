from __future__ import annotations

from typing import Any

import numpy as np


def binary_confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    ignore_value: int = 255,
) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError("Predizione e target devono avere la stessa forma.")
    valid = target != ignore_value
    valid &= (target == 0) | (target == 1)
    if not np.any(valid):
        return np.zeros((2, 2), dtype=np.int64)
    encoded = target[valid].astype(np.int64) * 2 + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=4).reshape(2, 2)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    if confusion.shape != (2, 2):
        raise ValueError("La matrice di confusione deve essere 2x2.")
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    intersection_over_union = ratio(tp, tp + fp + fn)
    dice = ratio(2 * tp, 2 * tp + fp + fn)
    total = tn + fp + fn + tp
    return {
        "valid_pixels": total,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": ratio(tn + tp, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "dice_coefficient": dice,
        "intersection_over_union": intersection_over_union,
        "positive_support": tp + fn,
        "negative_support": tn + fp,
    }


def render_confusion_matrix_svg(
    confusion: np.ndarray,
    output_path: str,
    *,
    title: str = "Matrice di confusione",
) -> None:
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())
    values = (tn, fp, fn, tp)
    maximum = max(values, default=0) or 1
    colors = ("#d9ead3", "#f4cccc", "#f4cccc", "#d9ead3")
    labels = ("TN", "FP", "FN", "TP")
    cells = []
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = 70 + (index % 2) * 170
        y = 90 + (index // 2) * 120
        opacity = 0.25 + 0.75 * value / maximum
        cells.append(
            f'<rect x="{x}" y="{y}" width="150" height="100" '
            f'fill="{color}" fill-opacity="{opacity:.3f}" stroke="#333"/>'
            f'<text x="{x + 75}" y="{y + 42}" text-anchor="middle" '
            f'font-size="16">{label}</text>'
            f'<text x="{x + 75}" y="{y + 72}" text-anchor="middle" '
            f'font-size="20">{value}</text>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="360" '
        'viewBox="0 0 420 360">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="210" y="35" text-anchor="middle" font-size="18">{title}</text>'
        '<text x="210" y="345" text-anchor="middle" font-size="13">'
        'Predizione: 0 / 1</text>'
        + "".join(cells)
        + "</svg>"
    )
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
