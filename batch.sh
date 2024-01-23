for i in {0..15};
do
python3 train_resnet.py --depth 2 --width 32 --fold $i;
done