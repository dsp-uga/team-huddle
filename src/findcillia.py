import argparse
import sys
import os
# This method will activate the argument parser and will have all the paramenters that will  be pased through the code
def get_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a Cilia detection network.')
    parser.add_argument('trainortest', help='Whether you want to train or test againts the data set', type=str)
    parser.add_argument('network', help='The network which you want to train, Default is FCN', default='FCN', type=str)
    parser.add_argument('--trainingdir', help='The Directory containing the Training Data', default="data/test/data",type=str)
    parser.add_argument('--testdir', help='The Directory containing the Testing Data.', default='data/train/data',type=str)
    parser.add_argument('--outputdir', help='The Directory for the output masks', default='output/', type=str)
    parser.add_argument('--masks', help='Path to a masks folder containing masks')
    parser.add_argument('--batchsize',      help='Size of the batches.', default=20, type=int)
    parser.add_argument('--noepoch', help='No of epochs you want to train', default=100, type=int)
    return parser.parse_args(args)



def main(args=None):
    print ("Cilia")
    if args is None:
        args = sys.argv[1:]
    args = get_args(args)
    if args.trainortest == "train": # first checks if we want to train or only test
        # Then depending on the model that model will be trained
        if args.network == "U-net":
            print("Training Unet")
        elif args.network == "FCN":
            print("Training FCN")
            os.system("python fcn.py 0 "+args.trainingdir+" "+args.testdir+" "+args.outputdir+" "+args.batchsize+ " "+args.noepoch)
        elif args.network == "opticalflow":
            print("Using optical-flow")
            os.system("python optical.py")
        else:
            print("Network not implimeneted")
            exit(0)
    elif args.trainortest == "test":
         # Then depending on the model that model will be trained
        if args.network == "U-net":
            print("Testing Unet")
        elif args.network == "FCN":
            print("Testing FCN")
            os.system("python fcn.py 1 " + args.trainingdir + " " + args.testdir + " " + args.outputdir + " " + args.batchsize + " " + args.noepoch)
        elif args.network == "opticalflow":
            print("Using optical-flow")
            os.system("python optical.py")
        else:
            print("Network not implimeneted")
            exit(0)
    else:
        print("Please mention either train or test")
        exit(0)


if __name__ == "__main__":
    main()
