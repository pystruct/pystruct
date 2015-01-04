#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

if [[ "$PYTHON_VERSION" == "3.4" ]]; then
    export PYTHON=python3
    export NOSETESTS=nosetests3
else
    export PYTHON=python
    export NOSETESTS=nosetests
fi

$PYTHON --version
$PYTHON -c "import numpy; print('numpy %s' % numpy.__version__)"
$PYTHON -c "import scipy; print('scipy %s' % scipy.__version__)"
$PYTHON -c "from pystruct.inference import get_installed; print('pystruct inference algorithms: %s' % get_installed())"


# Do not use "make test" or "make test-coverage" as they enable verbose mode
# which renders travis output too slow to display in a browser.
if [[ "$COVERAGE" == "true" ]]; then
    $NOSETESTS -s --with-coverage pystruct
else
    $NOSETESTS -s pystruct
fi

#make test-doc test-sphinxext
