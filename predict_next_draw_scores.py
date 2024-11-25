import json
import pandas as pd
import numpy as np
import logging
import argparse
from scipy.stats import t
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_json_input_into_df(input_json_path: str, months_filter: int = 36) -> pd.DataFrame:
    logging.info(f"Parsing JSON input from {input_json_path}")
    with open(input_json_path) as f:
        data = json.load(f)
    
    df = pd.DataFrame(columns=['draw_name', 'id', 'date', 'size', 'lowest_score'])

    for k, v in data.items():
        draw_df = pd.DataFrame.from_dict(data[k])
        draw_df['draw_name'] = k
        draw_df = draw_df.rename({'drawNumber': 'id', 'drawDate': 'date', 'drawSize': 'size', 'drawCRS': 'lowest_score'}, axis=1)
        draw_df = draw_df[['draw_name', 'id', 'date', 'size', 'lowest_score']]
        df = pd.concat([df, draw_df])

    df['size'] = df['size'].apply(lambda x: x.replace(',', ''))
    df = df.astype({"size": "int32", "lowest_score": "int32"})

    df["date"] = pd.to_datetime(df["date"])
    now = pd.Timestamp("now")
    months_filter = now - pd.DateOffset(months=months_filter)
    filtered_df = df[df['date'] >= months_filter]

    filtered_df.reset_index(drop=True, inplace=True)
    
    logging.info(f"Parsed DataFrame with {len(filtered_df)} records")
    return filtered_df


def get_draw_name_predictions(draw_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    logging.info(f"Building model for draw name: {draw_df['draw_name'].iloc[0]}")
    draw_df = draw_df.sort_values('date')
    draw_df['prev_lowest_score'] = draw_df['lowest_score'].shift(1)
    draw_df = draw_df.iloc[1:, :].reset_index(drop=True)
    
    model = LinearRegression()
    X = draw_df['prev_lowest_score'].to_numpy().reshape(-1, 1)
    y = draw_df['lowest_score'].to_numpy()
    model.fit(X, y)
    
    last_draw_lowest_score = draw_df['lowest_score'].iloc[-1]
    X_last_score = np.array([[last_draw_lowest_score]])
    y_pred = round(model.predict(X_last_score)[0], 2)

    y_pred_all = model.predict(X)
    residuals = y - y_pred_all
    RSS = np.sum(residuals**2)

    n = len(X)
    MSE = RSS / (n - 2)

    X_mean = np.mean(X)
    SE = np.sqrt(
        MSE * (1 + (1/n) + ((X_last_score[0][0] - X_mean) ** 2) / np.sum((X - X_mean) ** 2))
    )

    t_value = t.ppf(1 - alpha/2, df=n - 2)

    lower_bound = round(y_pred - t_value * SE, 2)
    upper_bound = round(y_pred + t_value * SE, 2)

    logging.info(f"""Last draw lowest score: {last_draw_lowest_score}, Next draw score prediction: {y_pred}, Confidence Interval: [{lower_bound}, {upper_bound}]""")
    
    draw_res = {
        'last_draw_lowest_score': float(last_draw_lowest_score),
        'prediction': float(y_pred),
        'confidence_interval': [float(lower_bound), float(upper_bound)],
        'standard_error': float(round(SE, 2))
    }
    return draw_res


def get_predictions(input_json_path: str, months_filter: int = 36, alpha: float = 0.05):
    df = parse_json_input_into_df(input_json_path, months_filter=months_filter)
    
    grouped_df = df[['draw_name', 'id']].groupby('draw_name').count().rename(columns={'id': 'count'})
    eligible_draw_names = grouped_df[grouped_df['count'] > 6].index.unique()
    results = {}
    
    for dn in eligible_draw_names:
        draw_df = df[df['draw_name'] == dn].copy()
        results[dn] = get_draw_name_predictions(draw_df, alpha=alpha)
    
    input_path_dir = '/'.join(input_json_path.split('/')[:-1])
    output_json_file = 'pred_' + input_json_path.split('/')[-1]
    output_json_path = input_path_dir + '/' + output_json_file
    
    with open(output_json_path, 'w') as fp:
        json.dump(results, fp)
    
    logging.info(f"Predictions saved to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict the next draw\'s lowest score for Express Entry.')
    parser.add_argument('input_json_path', type=str, help='Path to the input JSON file containing draw data.')
    parser.add_argument('--months_filter', type=int, default=36, help='Number of months to filter the draw data (default: 36).')
    parser.add_argument('--alpha', type=float, default=0.05, help='Confidence level for the prediction (default: 0.05).')
    
    args = parser.parse_args()
    
    logging.info("Starting prediction process")
    get_predictions(
        input_json_path=args.input_json_path,
        months_filter=args.months_filter,
        alpha=args.alpha
    )
    logging.info("Prediction process completed")
                                     