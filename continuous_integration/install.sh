#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

# add cython repository
sudo add-apt-repository ppa:cython-dev/master-ppa -y
sudo apt-get update -qq

if [[ "$OPENGM" == "true" ]]; then
    sudo add-apt-repository ppa:ukplc-team/testing -y
    sudo apt-get update -qq
    sudo apt-get install libhdf5-serial-dev libboost1.49-dev libboost-python1.49-dev
    git clone https://github.com/opengm/opengm.git
    cd opengm
    cmake . -DWITH_BOOST=TRUE -DWITH_HDF5=TRUE -DBUILD_PYTHON_WRAPPER=TRUE -DBUILD_EXAMPLES=FALSE -DBUILD_TESTING=FALSE
    make -j2 --quiet
    sudo make install
    cd ..
fi

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose cython scikit-learn cvxopt\
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    source activate testenv

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --yes mkl
    else
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # Use standard ubuntu packages in their default version
    # except for cython :-/
    sudo apt-get install -qq python-scipy python-nose python-pip python-cvxopt cython
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

# install our favorite inference packages 
if [[ "$INFERENCE" != "false" ]]; then
    pip install pyqpbo ad3 
fi

pip install scikit-learn

# Build scikit-learn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py build_ext --inplace
