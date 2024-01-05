# Computer Vision Homework 2


## Installation

> Note: You can use previous Conda environment for this homework.

To start your homework, you need to install requirements. We recommend that you use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment for this homework.

```bash
conda create -n cvhw2 python=3.7
conda activate cvhw2
```

You can install requirements with the following command in the homework directory:

```bash
pip install -r requirements.txt
```

In order to visualize plotly plots in jupyter notebooks, you should run the command given below.

```bash
conda install nodejs
```

Add your conda environment to Jupyter Notebook and enable widget rendering.

```bash
python -m ipykernel install --user --name=cvhw2
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable varInspector/main
```



## Submission

Any large-sized file (images, binaries, data files, etc) should be ignored while submitting the homework. Submissions are done via Ninova until the submission deadline. Submit the Jupyter notebook and report pdf separately. Do not edit ```utils.py``` or ```renderer.py``` since you will not submit these modules.


## Textbook

> [Concise Computer Vision
An Introduction
into Theory and Algorithms](https://doc.lagout.org/science/0_Computer%20Science/2_Algorithms/Concise%20Computer%20Vision_%20An%20Introduction%20into%20Theory%20and%20Algorithms%20%5BKlette%202014-01-20%5D.pdf)

## Contact

TA: Tolga Ok
okt@itu.edu.tr
