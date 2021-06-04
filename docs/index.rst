.. geology documentation master file, created by
   sphinx-quickstart on Tue Mar  3 13:38:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepField's documentation!
=====================================

`DeepField` is a machine learning framework for reservoir simulation.


Contents
========
.. toctree::
   :maxdepth: 4
   :titlesonly:

   api/deepfield


Supported keywords
==================

.. note:: Besides of standard keywords, any component supports
          custom keywords with array-like data.

General

* ARRA
* ARRAY
* DISGAS
* EFIL
* EFILE
* EFOR
* EFORM
* ETAB
* FIELD
* GAS
* HFIL
* HFILE
* HUNI
* HUNIS
* INCLUDE
* METRIC
* OIL
* START
* TITLE
* TFIL
* TTAB
* USERFILE
* VAPOIL
* WATER

Grid

* ACTNUM
* DIMENS
* DX
* DY
* DZ
* MAPAXES
* TOPS
* COORD
* DIMENS
* ZCORN

Rock

* PERMX
* PERMY
* PERMZ
* PORO

States

* PRESSURE
* SGAS
* SOIL
* SWAT
* RS

Tables

* PVTO
* PVDG
* PVTW
* SWOF
* SGOF
* RSVD
* ROCK
* DENSITY

Wells

* wells and groups related keywords from .RSM file
* control keywords from ETAB section
* COMPDAT
* GROU
* GROUP
* GRUPTREE
* PERF
* WCONPROD
* WCONINJE
* WELLTRACK
* WELSPECS
