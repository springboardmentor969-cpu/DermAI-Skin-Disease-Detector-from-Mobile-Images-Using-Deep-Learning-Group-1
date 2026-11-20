from __future__ import annotations

from pathlib import Path

import pytest

from app.ml.training import (
    classification_metrics,
    load_records,
    make_grouped_splits,
    make_transforms,
)
from app.ml.segmentation import (
    _write_new_text,
    discover_pairs,
    evaluate_segmentation_checkpoint,
    train_segmentation,
)
from app.services.image_quality_service import assess_image_quality
from app.services.lesion_detection_service import (
    get_lesion_detection_status,
    predict_lesion_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ham10000_records() -> list[dict[str, str | Path]]:
    return load_records(PROJECT_ROOT)


def test_ham10000_records_cover_real_dataset(
    ham10000_records: list[dict[str, str | Path]],
) -> None:
    records = ham10000_records
    assert len(records) == 10_015
    assert len({str(record["label"]) for record in records}) == 7
    assert all(Path(record["path"]).is_file() for record in records)


def test_grouped_splits_are_complete_disjoint_and_reproducible(
    ham10000_records: list[dict[str, str | Path]],
) -> None:
    records = ham10000_records
    first = make_grouped_splits(records)
    second = make_grouped_splits(records)
    assert first == second
    assert sorted(index for split in first.values() for index in split) == list(
        range(len(records))
    )
    groups = {
        name: {str(records[index]["lesion_id"]) for index in indices}
        for name, indices in first.items()
    }
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])
    expected_classes = {str(record["label"]) for record in records}
    for indices in first.values():
        assert {str(records[index]["label"]) for index in indices} == expected_classes
    split_by_image_hash: dict[str, set[str]] = {}
    for split_name, indices in first.items():
        for index in indices:
            image_hash = str(records[index]["image_hash"])
            split_by_image_hash.setdefault(image_hash, set()).add(split_name)
    assert all(len(splits) == 1 for splits in split_by_image_hash.values())


def test_grouped_split_rejects_mixed_diagnoses_for_one_lesion() -> None:
    records = [
        {"image_id": str(index), "lesion_id": str(index), "label": "nv", "path": Path("unused")}
        for index in range(20)
    ]
    records[1]["lesion_id"] = records[0]["lesion_id"]
    records[1]["label"] = "mel"
    with pytest.raises(ValueError, match="multiple diagnosis classes"):
        make_grouped_splits(records)


def test_classification_metrics_include_requested_measures_and_matrix() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["akiec", "bcc"])
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_avg"]["precision"] == pytest.approx(5 / 6)
    assert metrics["macro_avg"]["recall"] == pytest.approx(0.75)
    assert metrics["macro_avg"]["f1_score"] == pytest.approx(0.7333333333)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_image_quality_assessment_returns_measured_components() -> None:
    source_path = PROJECT_ROOT / "HAM10000_images_part_1" / "ISIC_0024306.jpg"
    image_bytes = source_path.read_bytes()
    result = assess_image_quality(image_bytes)
    assert 0 <= result["score"] <= 100
    assert result["method"] == "heuristic"
    assert result["metrics"]["width"] > 0
    assert set(result["components"]) == {
        "sharpness",
        "exposure",
        "contrast",
        "resolution",
        "artifacts",
    }
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert isinstance(result["feedback"], list)
    assert isinstance(result["recapture_recommended"], bool)


def test_image_quality_flags_blurred_underexposed_low_resolution_image() -> None:
    from PIL import Image
    from io import BytesIO

    image = Image.new("RGB", (128, 128), (10, 10, 10))
    payload = BytesIO()
    image.save(payload, format="PNG")
    result = assess_image_quality(payload.getvalue())
    assert result["category"] == "poor"
    assert result["recapture_recommended"] is True
    assert any("underexposed" in message for message in result["feedback"])
    assert any("resolution" in message for message in result["feedback"])


def test_quality_assessment_rejects_unreadable_data() -> None:
    with pytest.raises(ValueError, match="could not decode"):
        assess_image_quality(b"not an image")


def test_lesion_segmentation_reports_trained_artifact_status() -> None:
    status = get_lesion_detection_status()
    assert status["status"] == "available"
    assert "ISBI 2016" in status["reason"]


def test_trained_unet_returns_mask_and_overlay() -> None:
    source_path = PROJECT_ROOT / "HAM10000_images_part_1" / "ISIC_0024306.jpg"
    result = predict_lesion_mask(source_path.read_bytes())
    assert result["status"] == "available"
    assert result["architecture"] == "unet"
    assert str(result["mask_data_url"]).startswith("data:image/png;base64,")
    assert str(result["overlay_data_url"]).startswith("data:image/png;base64,")


def test_training_augmentation_and_evaluation_transforms_are_distinct() -> None:
    training, evaluation = make_transforms()
    assert training is not evaluation
    assert len(training.transforms) > len(evaluation.transforms)


def test_isbi_2016_segmentation_pairs_match_and_dimensions() -> None:
    dataset_root = PROJECT_ROOT / "segmentation_dataset"
    pairs = discover_pairs(dataset_root)
    assert len(pairs) == 900
    assert all(
        image.stem == mask.stem.removesuffix("_Segmentation")
        for image, mask in pairs
    )

    from PIL import Image

    assert all(
        Image.open(image).size == Image.open(mask).size
        for image, mask in pairs
    )


def test_segmentation_training_rejects_non_empty_output_directory(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "existing-run"
    output_directory.mkdir()
    (output_directory / "existing-report.json").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        train_segmentation(
            tmp_path / "missing-dataset",
            output_directory,
            epochs=1,
            batch_size=1,
        )
    assert (output_directory / "existing-report.json").read_text(
        encoding="utf-8"
    ) == "preserve"


def test_segmentation_output_preflight_allows_empty_directory(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "empty-run"
    with pytest.raises(FileNotFoundError):
        train_segmentation(
            tmp_path / "missing-dataset",
            output_directory,
            epochs=1,
            batch_size=1,
        )
    assert output_directory.is_dir()
    assert not any(output_directory.iterdir())


def test_segmentation_report_writer_accepts_encoding(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_new_text(report_path, '{"ok": true}', encoding="utf-8")
    assert report_path.read_text(encoding="utf-8") == '{"ok": true}'


def test_segmentation_evaluation_only_writes_metrics_without_training(
    tmp_path: Path,
) -> None:
    checkpoint = PROJECT_ROOT / "backend" / "artifacts" / "segmentation_isbi2016_run_20261102" / "unet.pth"
    output_directory = tmp_path / "evaluation"
    report = evaluate_segmentation_checkpoint(
        PROJECT_ROOT / "segmentation_dataset",
        checkpoint,
        output_directory,
        seed=42,
        batch_size=32,
    )
    assert report["evaluation_only"] is True
    assert report["split_counts"] == {"train": 630, "validation": 90, "test": 180}
    assert 0 <= report["models"]["unet"]["validation_dice"] <= 1
    assert 0 <= report["models"]["unet"]["test_dice"] <= 1
    assert checkpoint.is_file()
    assert all(
        (output_directory / filename).is_file()
        for filename in (
            "selection.json",
            "segmentation_report.json",
            "segmentation_report.md",
        )
    )


def test_segmentation_evaluation_refuses_existing_reports(tmp_path: Path) -> None:
    output_directory = tmp_path / "evaluation"
    output_directory.mkdir()
    (output_directory / "segmentation_report.json").write_text(
        "preserve", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        evaluate_segmentation_checkpoint(
            PROJECT_ROOT / "segmentation_dataset",
            PROJECT_ROOT
            / "backend"
            / "artifacts"
            / "segmentation_isbi2016_run_20261102"
            / "unet.pth",
            output_directory,
        )
    assert (output_directory / "segmentation_report.json").read_text(
        encoding="utf-8"
    ) == "preserve"
