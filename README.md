# Cilia Segmentation

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


## Authors

* **Ankita Joshi** - [AnkitaJo](https://github.com/AnkitaJo)
* **Parya Jandaghi** - [parya-j](https://github.com/parya-j)
* **Nihal Soans** - [nihalsoans91](https://github.com/nihalsoans91)

See also the list of [contributors](https://github.com/dsp-uga/team-huddle/blob/master/CONTRIBUTORS.md) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used


