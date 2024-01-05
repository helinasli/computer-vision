# Computer Vision Homework 3

## Installation

To start your homework, you need to install requirements. We recommend that you use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment for this homework. Run the below commands in the given order.

```bash
conda create -n cvhw3 python=3.7
conda activate cvhw3
```

You can install requirements with the following command in the homework directory:

```bash
pip install -r requirements.txt
```

In order to visualize canvas in jupyter notebooks, you should run the commands given below.

```bash
conda install nodejs
conda install -c conda-forge ipycanvas
```

Add your conda environment to Jupyter Notebook and enable widget rendering.

```bash
conda install -n cvhw3 ipykernel --update-deps --force-reinstall
jupyter nbextension enable --user --py widgetsnbextension
```

## Submission

Any large-sized file (images, binaries, data files, etc) should be ignored while submitting the homework. Submissions are done via Ninova until the submission deadline. Submit the Jupyter notebook ```hw3.ipynb``` and ```last_homework.py``` separately. Do not modeify ```visualizers``` since you will not submit that folder.

## Textbook

> [Concise Computer Vision
An Introduction
into Theory and Algorithms](https://doc.lagout.org/science/0_Computer%20Science/2_Algorithms/Concise%20Computer%20Vision_%20An%20Introduction%20into%20Theory%20and%20Algorithms%20%5BKlette%202014-01-20%5D.pdf)

## Contact

TA: Tolga Ok
okt@itu.edu.tr
