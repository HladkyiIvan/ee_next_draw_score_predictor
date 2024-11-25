# Express Entry Prediction Script

## Basic Usage

The `predict_next_draw_scores.py` script predicts the next draw's lowest score for Express Entry based on historical data.

### Command-Line Usage

```sh
predict_next_draw_scores.py [-h] [--months_filter MONTHS_FILTER] [--alpha ALPHA] input_json_path
```

### Description

Predict the next draw's lowest score for Express Entry.

### Arguments

- **positional arguments**:
  - `input_json_path`  
    Path to the input JSON file containing draw data.

- **optional arguments**:
  - `-h`, `--help`  
    Show this help message and exit.
  - `--months_filter MONTHS_FILTER`  
    Number of months to filter the draw data (default: 36).
  - `--alpha ALPHA`  
    Confidence level for the prediction (default: 0.05).

### Example

To run the script with an input JSON file and default parameters:

```sh
python predict_next_draw_scores.py data/express_entry_draws.json
```

To specify a different number of months for filtering and a custom confidence level:

```sh
python predict_next_draw_scores.py data/express_entry_draws.json --months_filter 24 --alpha 0.1
```

