#!/bin/bash

#    create env
conda create -n ufilm_segment python=3.6
conda activate ufilm_segment

#    install required packages
pip install git+https://github.com/dhlab-epfl/dhSegment.git
conda install tensorflow-gpu=1.13.1
pip install joblib
pip install sklearn
pip install numpy
pip install tqdm
# get ia utility, within a python env (avoid host-wide installation):
pip install internetarchive
git clone https://git.archive.org/merlijn/archive-hocr-tools
cd archive-hocr-tools/
python3 setup.py bdist_wheel && pip install -U dist/*.whl
cd ..
pip install ray==1.3.0
pip install pytest
pip install lxml
pip install Levenshtein
pip install pandas
