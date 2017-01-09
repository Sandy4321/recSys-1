# Recommender Systems

Recommender Systems Project
****************************

This folder contains the library and the entry points of the Recommender Systems project.

The code is organized in the following way
- **examples**: folder containing all the executable scripts that (may) exploit the *library* code
- **notebooks**: jupyter notebooks for analysis and reporting
- **datasources**: folder with original and partially processed data
- **library**: folder that collects all the low-level scripts

Datasources:
===========

High-level scripts:
-------------------

How to use:
===========

Installation
============

You can perform a minimal install of ``recSys`` with:

.. code:: shell

	git clone https://github.com/LeonardoCella/recSys.git
	cd recSys/source
	pip install -e .

Installing everything
---------------------

To install the whole set of features, you will need additional packages installed.
You can install everything by running ``pip install -e '.[all]'``.


What's new
----------

How to Import the Project in PyCharm
====================================
How to import the code in PyCharm
- File -> Open
- Select *source* folder
- Import the project
- On Project tab select root folder source, right click Mark directory as -> Sources root