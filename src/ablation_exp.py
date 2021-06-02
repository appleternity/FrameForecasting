from ml_baseline import *

def ablation_exp(block=200, model_name="RandomForest", data_name="bookcorpus", removed_dims=None, device="cpu", n_jobs=10):
   # tfidf as feature
    if data_name == "bookcorpus":
        x_train, y_train = load_tfidf("train", block, verbose=True)
        x_test, y_test = load_tfidf("test", block, verbose=True)
    elif data_name == "coda19":
        x_train, y_train = coda_load_tfidf("train", block, verbose=True)
        x_test, y_test = coda_load_tfidf("test", block, verbose=True)
    else:
        print("Not supported yet!")
        quit()

    x_train, y_train = x_train.todense(), y_train.todense()
    x_test, y_test = x_test.todense(), y_test.todense()

    print("Original Dimension")
    print("train: x = {}, y = {}".format(str(x_train.shape), str(y_train.shape)))
    print("test: x = {}, y = {}".format(str(x_test.shape), str(y_test.shape)))

    if removed_dims is None:
        removed_dims = []
    columns = [i for i in range(0, x_train.shape[1]) if i not in removed_dims]
    print(f"columns.size = {len(columns)}")
    x_train = x_train[:, columns]
    x_test = x_test[:, columns]

    print("After Frame Removal")
    print("train: x = {}, y = {}".format(str(x_train.shape), str(y_train.shape)))
    print("test: x = {}, y = {}".format(str(x_test.shape), str(y_test.shape)))
    print("building model using", model_name)
    
    # parameter setting
    rf_param = {
        "max_depth": 10, 
        "random_state": RANDOM_SEED, 
        "n_jobs": n_jobs, 
        "n_estimators": 30,
        "verbose": 10,
    }
    lgbm_param = {
        "max_depth": 3,
        "num_leaves": 5,
        "random_state": RANDOM_SEED,
        "n_estimators":100,
        "n_jobs": 1,
        "verbose": -1,
        "force_row_wise":True,
        "device": "gpu",
        "max_bin": 63,
    }
    if model_name == "RandomForest":
        model = RandomForestRegressor(**rf_param)
    elif model_name == "LGBM":
        model = MultiOutputRegressor(LGBMRegressor(**lgbm_param), n_jobs=n_jobs)
    else:
        print("Please use the available model")

    print("training")
    model.fit(x_train, y_train)

    print("prediting")
    print("block number = {}".format(block))
    y_pred = model.predict(x_test)
    res = tfidf_metric(y_test, y_pred, device=device)
    print("cosine", res)
    print_tfidf_metric(
        {
            "cosine": float(res),
            "block": block,
            "model": model_name,
            "note": "ablation - clean - tfidf",
            "dimension_removal": [int(d) for d in removed_dims],
        }, 
        filename=os.path.join(result_dir, f"ablation_exp_{data_name}_ml_baseline.json")
    )

def run_ablation_study():
    removed_dim_list = np.random.RandomState(234598).permutation(1221)

    # load finished terms
    data_name = "bookcorpus"
    filename = os.path.join(result_dir, f"ablation_exp_{data_name}_ml_baseline.json")
    
    if os.path.isfile(filename):
        with open(filename, 'r', encoding='utf-8') as infile:
            results = json.load(infile)
            for res in results:
                res["dimension_removal"]

            finished_dims = {
                d
                for res in results        
                for d in res["dimension_removal"]
            }
    else:
        finished_dims = set()

    print(finished_dims)
    for count, dim in enumerate(removed_dim_list[:]):
        if dim in finished_dims:
            continue
        start_time = datetime.now()
        print(f"\n\nStart Ablation Exp {count}-th with dim={dim} removal at {start_time}.")
        ablation_exp(block=150, model_name="LGBM", data_name="bookcorpus", removed_dims=[dim])
        end_time = datetime.now()
        print(f"Finish Ablation Exp with dim={dim} removal at {end_time}.")
        print(f"Took {(end_time-start_time).total_seconds()} Seconds!")

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study for ML Baseline.")
    parser.add_argument("--device", help="device used for computing tfidf [cpu/cuda:0/cuda:1]", type=str, default="cpu")
    parser.add_argument("--block", help="story block size", type=int, default=20)
    parser.add_argument("--data", help="Corpus used for training and testing [bookcorpus/coda19]", type=str, default="bookcorpus")
    parser.add_argument("--n_jobs", help="Processes used for computing", type=int, default=10)
    parser.add_argument("--model", help="ML model. [LGBM / RandomForest]", type=str, default="LGBM")
    parser.add_argument("--removed_dim", help="Dimention you want to remove.", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.removed_dim == None:
        run_ablation_study()
    else:
        ablation_exp(
            block=args.block, 
            model_name=args.model, 
            data_name=args.data, 
            n_jobs=args.n_jobs,
            device=args.device,
            removed_dims=[args.removed_dim],
        )

if __name__ == "__main__":
    main()
