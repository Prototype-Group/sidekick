# Sidekick (Beta) [![Build Status](https://travis-ci.com/Peltarion/sidekick.svg?token=nkS94uQqBVFyK1JitpGf&branch=master)](https://travis-ci.com/Peltarion/sidekick)

Sidekick is a package that helps you interact with features on the Peltarion platform. Sidekick aims to provide all functionality needed for solving machine learning projects end-to-end outside of and on the Pletarion platform. 

Currently, Sidekick provides:

* Data preprocessing. Create dataset splits, remove missing values, imputation, and more
* Data inspection for data exploration (histogram, correlation plot, etc)
* [Convert your dataset to work with the Platform](#make-your-data-have-a-format-that-is-supported-on-the-platform)
* [Upload your prepared data to the Platform](#upload-dataset-through-the-peltarion-platforms-data-api)
* [Make predictions with models you have trained on the Platform](#make-predictions-with-a-trained-and-deployed-model-on-the-platform)

If you experience any issues or feel that features are missing from Sidekick, you can always open an issue [here](https://github.com/Peltarion/sidekick/issues/new/choose)

## Getting started with Python

If you are unfamiliar with Python and how to install it, we recommend that you install Anaconda, a method that simplifies your Python installation:

* [Install Anaconda](https://docs.anaconda.com/anaconda/install/)

If you just want to install python, you can follow one of the guides listed here:

* [Setting up Python3](https://docs.python-guide.org/starting/installation/#python-3-installation-guides)
* [Learning Python](https://realpython.com/python-first-steps/#how-to-download-and-install-python)

## Installation of sidekick

**Requirements** Sidekick requires python version 3.5 or newer.

If you are unfamiliar with python and how to install python, we recommend you follow the section [Getting started with python](#getting-started-with-python)

We also recommend you create a virtual environment and install Sidekick there. This is a general best practice in python and ensures that only the packages we need are installed and with the correct versions.

Here are some guides for setting up and activating a virtual environment. These guides assume that you have installed Anaconda or python and are running these commands in a terminal:

* [Creating a virtual environment](https://realpython.com/python-virtual-environments-a-primer/#create-it)
* [Activating the virtual environment](https://realpython.com/python-virtual-environments-a-primer/#activate-it)

Once the virtual environment is activated you can install Sidekick directly from Github:

```shell
pip install git+https://github.com/Peltarion/sidekick#egg=sidekick
```

# Usage

Examples of how to use sidekick are available at [examples/](examples/).
To start the notebooks (ends with extension .ipynb), you will need to install Jupyter notebook and run those files.

If you have installed Anaconda, then Jupyter notebook is already installed and can be started by executing the following command in your terminal:

```shell
jupyter notebook
```

If this command is not recognized, you can install Jupyter notebook by following one of these guides:

* https://jupyter.org/install
* https://docs.jupyter.org/en/latest/install/notebook-classic.html


# Features of Sidekick
## Make your data have a format that is supported on the platform

[See: Check if dataset has the correct format page](./pages/01_check_data_format.md)

## Upload dataset through the Peltarion Platform's Data API

[See: How to upload data to the Platform](./pages/02_upload_data_to_platform.md)

## Make predictions with a trained and deployed model on the Platform

[See: Make prediction to a deployed model on platform](./pages/03_make_prediction_tp_deployed_model.md)
