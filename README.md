[![Build Status](https://travis-ci.org/pystruct/pystruct.png)](https://travis-ci.org/pystruct/pystruct)
[![pypi version](http://img.shields.io/pypi/v/pystruct.svg?style=flat)](https://pypi.python.org/pypi/pystruct/)
[![pypi downloads](http://img.shields.io/pypi/dm/pystruct.svg?style=flat)](https://pypi.python.org/pypi/pystruct/)
[![licence](http://img.shields.io/badge/licence-BSD-blue.svg?style=flat)](https://github.com/pystruct/pystruct/blob/master/LICENSE)


PyStruct
========

PyStruct aims at being an easy-to-use structured learning and prediction library.
Currently it implements only max-margin methods and a perceptron, but other algorithms
might follow.

The goal of PyStruct is to provide a well-documented tool for researchers as well as non-experts
to make use of structured prediction algorithms.
The design tries to stay as close as possible to the interface and conventions
of [scikit-learn](http://scikit-learn.org).

You can install pystruct using

> pip install pystruct

Some of the functionality (namely OneSlackSSVM and NSlackSSVM) requires that cvxopt is installed.
See the [installation instructions](http://pystruct.github.io/intro.html) for more details.

The full documentation and installation instructions can be found at the website:
http://pystruct.github.io

You can contact the authors either via the [mailing list](https://groups.google.com/forum/#!forum/pystruct)
or on [github](https://github.com/pystruct/pystruct).

Currently the project is mostly maintained by Andreas Mueller, but contributions are very welcome.
