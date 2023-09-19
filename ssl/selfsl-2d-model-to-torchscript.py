import sys
sys.path.append(r"..")

if __name__ == "__main__":
    train_loader_call = None
    from lib.entrypoints.ssl.model_to_torchscript import main
    main()