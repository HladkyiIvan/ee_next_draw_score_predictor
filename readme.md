## Basic usage:

predict_next_draw_scores.py [-h] [--months_filter MONTHS_FILTER] [--alpha ALPHA] input_json_path

Predict the next draw's lowest score for Express Entry.

positional arguments:
  input_json_path       Path to the input JSON file containing draw data.

optional arguments:
  -h, --help            show this help message and exit
  --months_filter MONTHS_FILTER
                        Number of months to filter the draw data (default: 36).
  --alpha ALPHA         Confidence level for the prediction (default: 0.05).