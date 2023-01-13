import argparse
import json
import pandas

desc = """
Converts a CSV into a JSON file.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path',dest='path',help="Path to CSV",required=True)
    parser.add_argument(
        '--id_columns',dest='id_columns',nargs='+',
        help="Column with IDs",required=True)
    parser.add_argument(
        '--feature_columns',dest='feature_columns',nargs='+',
        help='columns with features',required=True)

    args = parser.parse_args()

    data = pandas.read_csv(args.path)
    data_ids = data[args.id_columns].astype(str)
    print(data.columns)
    data_features = data[args.feature_columns]
    output = {}

    for i in range(data.shape[0]):
        patient_id = '_'.join(data_ids.iloc[i].tolist())
        features = {}
        for f in args.feature_columns:
            v = data_features[f].iloc[i].tolist()
            features[f] = v
        output[patient_id] = features
    
    print(json.dumps(output,indent=4))