# Cilia Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The task is to design an algorithm that learns how to segment cilia. Cilia are microscopic
hairlike structures that protrude from literally every cell in your body. They beat
in regular, rhythmic patterns to perform myriad tasks, from moving nutrients in to moving
irritants out to amplifying cell-cell signaling pathways to generating calcium fluid
flow in early cell differentiation. Cilia, and their beating patterns, are increasingly being
implicated in a wide variety of syndromes that affected multiple organs.

## Getting Started

If you follow the below instructions it will allow you to install and run the training or testing.

### Prerequisites

What things you need to install the software and how to install them

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization so that you dont mess up your system environment
- [Keras](https://keras.io/) The best Deep Learning Tool PERIOD ;)
- [Tensorflow](https://www.tensorflow.org/) One of the API used as Backend of Keras

### Installing

#### Anaconda

Anaconda is a complete Python distribution embarking automatically the most common packages, and allowing an easy installation of new packages.

Download and install Anaconda from (https://www.continuum.io/downloads).
The link for Linux,Mac and Windows are in the website.Following their instruction will install the tool.
##### Running Environment

* Once Anaconda is installed open anaconda prompt(Windows/PC) Command Line shell(Mac OSX or Unix)
* Run ```conda env create -f environment.yml``` will install all packages required for all programs in this repository
###### To start the environment 

* For Unix like systems ```source activate cilia-env```

* For PC like systems ```activate cilia-env```

#### Keras

You can install keras using ``` pip ``` on command line
``` sudo pip install keras ```

The `environment.yml` file for conda is placed in [Extra](https://github.com/dsp-uga/team-huddle/tree/master/extra) for your ease of installation this has keras

#### Tensorflow
Installing Tensorflow is straight forward using ``` pip ``` on command line

* If CPU then  ``` sudo pip install tensorflow ```
* If GPU then ``` sudo pip install tensorflow-gpu ```

The `environment.yml` file for conda is placed in [Extra](https://github.com/dsp-uga/team-huddle/tree/master/extra) for your ease of installation this has tensorflow

#### Downloading the dataset (Optional)

If you prefer to download the dataset rather than online
The code is present in extra/downloadfiles.py

To Run ``` python downloadfiles.py ``` This will download the whole data set including training and testing

In Folders ```\Train``` and ```\Test``` respectively

## Data
The data itself are grayscale 8-bit images taken with DIC optics of cilia biopsies published
in this 2015 study. For each video, you are provided 100 subsequent frames, which
is roughly equal to about 0.5 seconds of real-time video (the framerate of each video is
200 fps). Since the videos are grayscale, if you read a single frame in and notice its data
structure contains three color channels, you can safely pick one and drop the other two.
Same goes for the masks.
Speaking of the masks: each mask is the same spatial dimensions (height, width) as the
corresponding video. Each pixel, however, is colored according to what it contains in the
video:
* 2 corresponds to cilia (what you want to predict!)
* 1 corresponds to a cell
* 0 corresponds to background (neither a cell nor cilia)

For more information please refer to our [wiki](https://github.com/dsp-uga/team-huddle/wiki) on [data](https://github.com/dsp-uga/team-huddle/wiki/Data)

## Running and Training

One can run `findcilia.py` via regular **python** 

```
$ python findcilia.py [train or Test] [Network] [optional args]
```
Example: ```python findcilia.py train FCN ```

  - **Required Arguments**

    - `trainortest`: This is a string either train or test

    - `network`: String which defines which network you want ot train or test Eg: FCN ,U-net,Tiramisu

  - **Optional Arguments**

    - `-batch-size`: The batch size if applicable (Default: `20`)
    - `-masks`: Path to the masks directory where masks are present. (Default: `train\masks`)
    - `-dataset`: Path to the dataset directory where train dataset is present. (Default: `train\`)


## Results

## Authors

* **Ankita Joshi** - [AnkitaJo](https://github.com/AnkitaJo)
* **Parya Jandaghi** - [parya-j](https://github.com/parya-j)
* **Nihal Soans** - [nihalsoans91](https://github.com/nihalsoans91)

See also the list of [contributors](https://github.com/dsp-uga/team-huddle/blob/master/CONTRIBUTORS.md) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments and References

* Hat tip to anyone who's code was used
* The project4 description used in Data Science Practicum [pdf](https://github.com/dsp-uga/sp18/blob/master/projects/p4/project4.pdf)
* An implementation of Fully Convolutional Networks with Keras [link](https://github.com/JihongJu/keras-fcn)


