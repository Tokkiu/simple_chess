simple_chess
===============================

.. image:: https://travis-ci.org/tokkiu/simple_chess.png
   :target: https://travis-ci.org/tokkiu/simple_chess
   :alt: Build Status

.. image:: https://landscape.io/github/tokkiu/simple_chess/master/landscape.png
   :target: https://landscape.io/github/tokkiu/simple_chess/master
   :alt: Code Health

.. image:: https://coveralls.io/repos/tokkiu/simple_chess/badge.png
   :target: https://coveralls.io/r/tokkiu/simple_chess
   :alt: Coverage Status

A simple implementation for simple chess

* Free software: MIT license
* Documentation: http://simple_chess.rtfd.org/

Data Structure
--------
Use 8*8 matrix to represent the state of the chessborad, e.g.,
::
 [[ 1.  1.  0.  0. -1. -1. -1. -1.]
 [ 1.  1.  1. -1.  1.  1. -1.  1.]
 [-1. -1.  1. -1.  1.  1.  1. -1.]
 [-1.  1. -1.  0.  1. -1.  1. -1.]
 [-1.  1. -1.  1.  1.  0.  1. -1.]
 [-1.  1. -1.  1. -1. -1. -1.  1.]
 [ 0.  1. -1. -1.  1.  0.  0. -1.]
 [ 1.  0.  1. -1.  1. -1.  1.  0.]]

The first hand is 1, and the second hand is -1, 0 is the positions haven't been placed.

Features
--------

TODO

Install
-------

::

   pip install simple_chess

Examples
--------

TODO

Changelog
---------

0.0.1
~~~~~~~~~~~~~~~~~~~~~~~~~~

2021-01-01

* init

Contributing
------------

::

   git clone git@github.com:tokkiu/simple_chess.git
   cd simple_chess
   virtualenv .
   make


Releasing
------------

::

   git clone git@github.com:tokkiu/simple_chess.git
   cd simple_chess
   pip install wheel
   python setup.py sdist bdist_wheel

Credits
-------

* Ary <mrgao.ary@gmail.com>
* Sarah <shenruhui@outlook.com>

Add your name and email, detail in: https://github.com/tokkiu/simple_chess/graphs/contributors

.. image:: https://d2weczhvl823v0.cloudfront.net/tokkiu/simple_chess/trend.png
   :alt: Bitdeli badge
   :target: https://bitdeli.com/free
