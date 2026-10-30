"""An executable offline example; inputs are explicitly synthetic."""

from synthetic_market.data import make_demo_data
from synthetic_market.model import ModelConfig, fit_market
from synthetic_market.validation import evaluate_holdout


def main():
    data = make_demo_data(n_obs=1000, seed=1729)
    model = fit_market(data, ModelConfig(mean="zero"))
    scenarios = model.simulate(n_paths=100, horizon=21, seed=42)
    print(scenarios.to_frame().head())
    split = str(data.close.index[799].date())
    report = evaluate_holdout(data, split, seed=7)
    print("Model:", report["metrics_model"])
    print("Unfiltered baseline:", report["metrics_baseline"])
    print("This example tests the software on a known synthetic process, not real markets.")


if __name__ == "__main__":
    main()
