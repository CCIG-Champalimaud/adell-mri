import sys
sys.path.append(r"..")

if __name__ == "__main__":
    train_loader_call = None
    from lib.entrypoints.ssl.train_2d import main
    main(sys.argv[1:])