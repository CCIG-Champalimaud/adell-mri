import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path")
    parser.add_argument("--factor_column",default="use_case_form")
    parser.add_argument("--id_column",default="study_uid")
    parser.add_argument("--factors_for_supervision",nargs="+",default=[])
    parser.add_argument("--fraction_supervision",default=0.2,type=float)
    parser.add_argument("--seed",default=42,type=int)
    parser.add_argument("--output_ssl_ids",required=True)
    parser.add_argument("--output_supervision_ids",required=True)

    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)
    if "prospective" in df:
        df = df[df["prospective"] == False]
    df = df[[args.id_column,args.factor_column]].drop_duplicates()
    total_n = df.shape[0]
    supervision_n = int(np.round(df.shape[0] * args.fraction_supervision))
    ssl_n = total_n - supervision_n

    ids = df[args.id_column]
    factor = df[args.factor_column]

    factors_for_supervision = [
        x.encode().decode('unicode_escape')
        for x in args.factors_for_supervision]

    factors_idx = np.zeros(df.shape[0],dtype=bool)
    for f in factors_for_supervision:
        factors_idx[factor == f] = True

    missing_supervision_n = supervision_n - factors_idx.sum()
    if missing_supervision_n > 0:
        rng = np.random.default_rng(args.seed)
        print(missing_supervision_n)
        add_idxs = rng.choice(
            np.where(ids[factors_idx == False])[0],
            missing_supervision_n,replace=False)
        factors_idx[add_idxs] = True
    
    supervision_ids = ids[factors_idx]
    ssl_ids = ids[~factors_idx]

    with open(args.output_ssl_ids,"w") as o:
        o.write("\n".join(ssl_ids))
    
    with open(args.output_supervision_ids,"w") as o:
        o.write("\n".join(supervision_ids))