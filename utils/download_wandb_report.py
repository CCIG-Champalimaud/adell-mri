import argparse
import pandas as pd 
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",dest="input_path",required=True)
    parser.add_argument(
        "--output_path",dest="output_path",required=True)
    args = parser.parse_args()
    
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(args.input_path)

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    o = {"name":[],"key":[],"value":[]}
    for c,n in zip(summary_list,name_list):
        for k in c:
            o["name"].append(n)
            o["key"].append(k)
            o["value"].append(c[k])
    runs_df = pd.DataFrame.from_dict(o)

    runs_df.to_csv(args.output_path)
