from eduguard.data import synthetic_data, FEATURES
from eduguard.model import train, load_bundle, infer, predict_sets
def test_demo_training(tmp_path):
    df = synthetic_data(n=500)
    result = train(df, directory=tmp_path, demo=True)
    assert result["dataset"] == "synthetic demo"
    bundle = load_bundle(tmp_path)
    p, s = infer(bundle, df[FEATURES].head(5))
    assert len(p) == 5 and s.shape == (5,2)
    assert all(0 <= x <= 1 for x in p)
def test_prediction_sets():
    sets = predict_sets(__import__("numpy").array([0.1, 0.9]), 0.8)
    assert sets.shape == (2,2)
