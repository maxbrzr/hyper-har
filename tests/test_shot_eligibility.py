from baya_har.experiments.check_shot_eligibility import assess_class_counts


def test_assess_class_counts_requires_support_and_query_for_every_class() -> None:
    rows = assess_class_counts(
        {0: 20, 1: 17, 2: 5},
        k_values=(0, 4, 5, 16),
        min_query_per_class=1,
    )

    assert [row["valid"] for row in rows] == [True, True, False, False]
    assert rows[2]["insufficient_class_counts"] == {2: 5}
    assert rows[3]["insufficient_class_counts"] == {2: 5}


def test_assess_class_counts_rejects_invalid_inputs() -> None:
    try:
        assess_class_counts({}, (1,))
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty class counts should fail")

    try:
        assess_class_counts({0: 2}, (-1,))
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative K should fail")
