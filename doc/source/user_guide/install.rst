####################################
Install SMRT
####################################

Option 1: Install the latest stable SMRT package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To get started with SMRT, you will need to install the latest stable version.

1.  **Install directly from the repository**:
    .. code:: shell
        pip install smrt

If you need features that are still under development, you may want to install the latest developers' version of SMRT

Option 2: Using `pip` for development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you need features that are still under development, you may want to install the latest developers' version of SMRT.

Make sure you have activated a virtual environment so that you do not get conflicts with another installed version of smrt. This can be done for example with `venv` but please refer to https://docs.python.org/3/library/venv.html if this is new to you. Most IDE have their own way of generating virtual environments, which may be easier than using `venv`.

.. code:: shell

   pip install git+https://github.com/smrt-model/smrt