AulePredictor
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RaulFD-creator/aulepredictor/workflows/CI/badge.svg)](https://github.com/RaulFD-creator/aulepredictor/actions?query=workflow%3ACI)


Computational tool for the prediction of metal-binding sites in proteins using deep convolutional neural networks. 

## Data generation
To generate a data set for model training the data generation pipeline can be easily executed with the pipeline.py script. 

## Installation
To install AulePredictor it is recommended to prepare a conda environment. 

`git clone https://github.com/RaulFD-creator/aulepredictor`

`pip install -e aulepredictor`

## Use
To use AulePredictor with default settings simply introduce de PDB ID of the protein.

`aulepredictor 1dd9`

To define where the file is to be stored:

`aulepredictor 1dd9 ./outputs/`

### Copyright

Copyright (c) 2022, Raúl Fernández Díaz


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
