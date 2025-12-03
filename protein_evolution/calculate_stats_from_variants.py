import pandas as pd
import sys

PREDICTION_PATH = ""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        PREDICTION_PATH = sys.argv[1]
    else:
        print("Error: PREDICTION_PATH must be set. Pass it as an argument.")
        print("Usage: python calculate_stats_from_variants.py [prediction_path]")
        sys.exit(1)

    data = pd.read_csv(PREDICTION_PATH)

    print(f'Calculating statistics from this path: {PREDICTION_PATH}')
    print(f'The average fitness over all {data.shape[0]} variants is {data["fitness"].mean():.4f}')
    print(f'Count of variants across dataset_used (either surrogate or DMS): {data["dataset_used"].value_counts()}')