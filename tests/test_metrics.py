from amharic_sentiment.metrics import bootstrap_macro_f1_ci, classification_metrics


def test_classification_metrics_use_fixed_three_class_order():
    result = classification_metrics(
        [0, 1, 2, 2],
        [0, 2, 2, 1],
        ["positive", "neutral", "negative"],
    )
    assert result["accuracy"] == 0.5
    assert len(result["confusion_matrix"]) == 3
    assert result["per_class"]["positive"]["support"] == 1
    assert result["per_class"]["neutral"]["support"] == 1
    assert result["per_class"]["negative"]["support"] == 2


def test_bootstrap_macro_f1_ci_is_reproducible_and_bounded():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    first = bootstrap_macro_f1_ci(y_true, y_pred, 3, n_bootstrap=100, seed=7)
    second = bootstrap_macro_f1_ci(y_true, y_pred, 3, n_bootstrap=100, seed=7)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0
