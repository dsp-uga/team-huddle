import argparse
import sys
# This method will activate the argument parser and will have all the paramenters that will  be pased through the code
def get_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a Cilia detection network.')
    parser.add_argument('trainortest', help='Whether you want to train or test againts the data set', type=str)
    parser.add_argument('network', help='Backbone model used by retinanet.', default='FCN', type=str)
    parser.add_argument('--dataset', help='Path to dataset folder for training.')
    parser.add_argument('--masks', help='Path to a masks folder containing masks')
    parser.add_argument('--batch-size',      help='Size of the batches.', default=20, type=int)

    return parser.parse_args(args)



def main(args=None):
    print ("Cilia")
    if args is None:
        args = sys.argv[1:]
    args = get_args(args)
    if args.trainortest == "train":
        if args.network == "U-net":
            print("Training Unet")
        elif args.network == "FCN":
            print("Training FCN")
        elif args.network == "tiramisu":
            print("training tiramisu")
        else:
            print("Network not implimeneted")
            exit(0)
    elif args.trainortest == "test":
        if args.network == "U-net":
            print("Testing Unet")
        elif args.network == "FCN":
            print("Testing FCN")
        elif args.network == "tiramisu":
            print("testing tiramisu")
        else:
            print("Network not implimeneted")
            exit(0)
    else:
        print("Please mention either train or test")
        exit(0)


if __name__ == "__main__":
    main()
