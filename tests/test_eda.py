import pandas as pd

from src.common import Paths, read_json
from src.data.cleaning import clean_results
from src.data.eda import run
from src.demo import synthetic_raw


def test_eda_tables_and_figures_reconcile(tmp_path):
    paths = Paths(tmp_path)
    clean, _ = clean_results(synthetic_raw())
    paths.interim.mkdir(parents=True)
    clean.to_parquet(paths.interim / "results_clean.parquet", index=False)
    run(paths)
    root = paths.reports / "eda"
    target = pd.read_csv(root / "tables/target_summary.csv").iloc[0]
    report = read_json(root / "eda_report.json")
    assert target.n_events == report["n_events"]
    assert target.n_positive == report["n_positive"]
    assert abs(target.positive_rate - report["positive_rate"]) < 1e-12
    assert len(list((root / "tables").glob("*.csv"))) == 9
    assert len(list((root / "figures").glob("*.png"))) == 6
