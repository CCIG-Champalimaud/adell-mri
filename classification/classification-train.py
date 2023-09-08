import sys
sys.path.append(r"..")

if __name__ == "__main__":
    train_loader_call = None
    from lib.entrypoints.classification.train import main
    main()