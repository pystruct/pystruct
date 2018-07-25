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
export PIP=pip

# add cython repository
#sudo add-apt-repository ppa:cython-dev/master-ppa -y
#sudo apt-get update -qq

if [[ "$OPENGM" == "true" ]]; then
    git clone https://github.com/opengm/opengm.git
    cd opengm
    # old cmake . -DCMAKE_INSTALL_PREFIX=/home/travis/.local -DWITH_BOOST=TRUE -DWITH_HDF5=TRUE -DBUILD_PYTHON_WRAPPER=TRUE -DBUILD_EXAMPLES=FALSE -DBUILD_TESTING=FALSE
    # old make -j2 --quiet
	cmake . -DCMAKE_INSTALL_PREFIX=/home/travis/.local -DWITH_BOOST=TRUE -DWITH_HDF5=TRUE -DWITH_AD3=FALSE -DWITH_TRWS=FALSE  -DWITH_QPBO=FALSE -DWITH_MRF=FALSE  -DWITH_GCO=FALSE  -DWITH_CONICBUNDLE=FALSE  -DWITH_MAXFLOW=FALSE  -DWITH_MAXFLOW_IBFS=FALSE -DBUILD_PYTHON_WRAPPER=TRUE -DBUILD_COMMANDLINE=FALSE -DCI=TRUE
    make -j1 --quiet
    make install
    cd ..
fi

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget https://repo.continuum.io/miniconda/Miniconda2-4.3.31-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda2
    export PATH=$HOME/miniconda2/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose cython\
        scikit-learn cvxopt pytest future \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION

    source activate testenv

elif [[ "$DISTRIB" == "conda3" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda3
    export PATH=$HOME/miniconda3/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose cython\
        scikit-learn cvxopt pytest future \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION

    source activate testenv

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # Use standard ubuntu packages in their default version
    # except for cython :-/
    $PIP install --user cvxopt
    $PIP install --user future  # for AD3
fi

if [[ "$COVERAGE" == "true" ]]; then
    $PIP install coverage coveralls
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
# install our favorite inference packages 
# Need  Transkribus/AD3  for now  $PIP install pyqpbo ad3 scikit-learn
$PIP install pyqpbo scikit-learn

# Build scikit-learn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py build_ext --inplace

#get Transkribus/AD3
git clone https://github.com/andre-martins/AD3
pushd AD3
python setup.py install
popd
python -c "import ad3; print(ad3.__version__)"
