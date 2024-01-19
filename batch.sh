python3 train_resnet.py --depth 2 --width 32 --project 2 &&
python3 train_resnet.py --depth 3 --width 32 --project 3 &&
python3 train_resnet.py --depth 1 1 --width 32 64 --project 4 &&
python3 train_resnet.py --depth 2 1 --width 32 64 --project 5 &&
python3 train_resnet.py --depth 1 2 --width 32 64 --project 6 &&
python3 train_resnet.py --depth 1 1 1 --width 32 64 128 --project 7 &&
python3 train_resnet.py --depth 1 1 2 --width 32 64 128 --project 8 &&
python3 train_resnet.py --depth 1 1 2 1 --width 32 64 128 256 --project 9 &&
python3 train_resnet.py --depth 1 1 2 2 --width 32 64 128 256 --project 10