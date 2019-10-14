import argparse
import pandas as pd
import numpy as np
import json


def normalize(args):
    cases=np.loadtxt(args.case_file, delimiter=",")
    print("cases:", cases.shape)

    if args.control_file is not None:
        controls=np.loadtxt(args.control_file, delimiter=",")
        print("controls:", controls.shape)
        
        if args.balance:
            # balance the data: there are fewer controls
            n = min(cases.shape[0], controls.shape[0])
            cases=cases[0:n]
            controls=controls[0:n]
        
        #normalize
        X_df=pd.concat([pd.DataFrame(cases),pd.DataFrame(controls)])
        print("X_df:", X_df.shape)
        
        X_df = X_df.applymap(lambda x: np.log(x + 1))
        mean = np.mean(X_df.values)
        std = np.std(X_df.values)
        with open(args.stats_file, "w+") as f:
            f.write(json.dumps({"mean": mean, "std": std}))

        normalize_df = X_df.applymap(lambda x: (x-mean)/std)

        Xtarget=np.array([0]*cases.shape[0] + [1]*controls.shape[0])
    
        normalize_df["case_control"] = Xtarget
        print("normalize_df:", normalize_df.shape)
        normalize_df.to_csv(args.output_file,index=False)
    else:
        #normalize
        X_df=pd.DataFrame(cases[:,:-args.one_hot_columns])
        XTarget = pd.DataFrame(cases[:,-args.one_hot_columns:])
        print("X_df:", X_df.shape)
        
        X_df = X_df.applymap(lambda x: np.log(x + 1))
        mean = np.mean(X_df.values)
        std = np.std(X_df.values)
        with open(args.stats_file, "w+") as f:
            f.write(json.dumps({"mean": mean, "std": std}))

        normalize_df = X_df.applymap(lambda x: (x-mean)/std)

        normalize_df = pd.DataFrame(np.concatenate([normalize_df.values, XTarget.values], axis=1))
        print("normalize_df:", normalize_df.shape)
        normalize_df.to_csv(args.output_file,index=False)
        


def discretize(x):
    if x < 0.5:
        return 0
    else:
        return 1
    

def denormalize(args):
    df=pd.read_csv(args.input_file, header=None)
    print("input:", df.shape)

    case_or_control_series0 = df.iloc[:, -args.one_hot_columns:]

    print("one_hot_column:", case_or_control_series0.shape)
    if args.discretize:
        Xtarget = case_or_control_series0.apply(discretize)
    else:
        Xtarget = case_or_control_series0

    print(args.one_hot_columns)
    X_df = df.iloc[:, :-args.one_hot_columns]

    with open(args.stats_file) as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std"]
    
    print("X_df:", X_df.shape)
    
    X_df = X_df.applymap(lambda x: x * std + mean)
    denormalize_df = X_df.applymap(lambda x: np.exp(x) - 1)
    
    denormalize_df = pd.concat([denormalize_df, Xtarget], axis=1)
    print("denormalize_df:", denormalize_df.shape)
    denormalize_df.to_csv(args.output_file,index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers()

    parser_normalize = subparsers.add_parser("normalize")
    parser_normalize.add_argument("--case_file", type=str,required=True)
    parser_normalize.add_argument("--control_file", type=str,required=False)
    parser_normalize.add_argument("--one_hot_columns", type=int,required=False)
    parser_normalize.add_argument("--output_file", type=str,required=True)
    parser_normalize.add_argument("--stats_file", type=str, required=True)
    parser_normalize.add_argument("--balance", action="store_true")
    parser_normalize.set_defaults(func=normalize)
    
    parser_denormalize = subparsers.add_parser("denormalize")
    parser_denormalize.add_argument("--input_file", type=str,required=True)
    parser_denormalize.add_argument("--output_file", type=str,required=True)
    parser_denormalize.add_argument("--stats_file", type=str, required=True)
    parser_denormalize.add_argument("--one_hot_columns", type=int,default=1, required=False)
    parser_denormalize.add_argument("--discretize", action="store_true")
    parser_denormalize.set_defaults(func=denormalize)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
