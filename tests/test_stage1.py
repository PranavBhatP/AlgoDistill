"""Tests for Stage 1 Tag Classifier."""

import pytest
from src.data.dataset_builder import DatasetBuilder
from src.models.stage1_classifier import Stage1TagClassifier, Stage1Config


def test_stage1_classifier_training_and_inference():
    """Verify Stage 1 classifier trains and predicts properly formatted probabilities."""
    builder = DatasetBuilder()
    dataset = builder.build_raw_dataset()
    train, val, test = builder.create_stratified_splits(dataset)

    clf = Stage1TagClassifier(Stage1Config(mode="fast"))
    train_res = clf.train(train, val)
    assert train_res["train_accuracy"] > 0.80

    report = clf.evaluate(test)
    assert report.top_3_accuracy >= 0.50
    assert report.total_samples == len(test)

    pred = clf.predict("Find the shortest path in a graph using Dijkstra priority queue.")
    assert "predicted_tag" in pred
    assert "confidence" in pred
    assert "top_k" in pred
    assert len(pred["top_k"]) == 3
    assert 0.0 <= pred["confidence"] <= 1.0
