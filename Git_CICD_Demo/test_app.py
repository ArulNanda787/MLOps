from app import predict_label


def test_positive_prediction():
    assert predict_label(0.8) == "negative"


def test_negative_prediction():
    assert predict_label(0.2) == "negative"
