# fast sinc transform

A python implementation of the fast sinc-transform described by [Greengard et. al., 2006](https://doi.org/10.2140/camcos.2006.1.121) as implemented by Hannah Lawrence in [fast-sinc-transform](https://fast-sinc-transform.readthedocs.io/en/latest/Overview.html). Utilizes [FINUFFT](https://finufft.readthedocs.io/) and [fastgl](https://people.sc.fsu.edu/~jburkardt/py_src/fastgl/fastgl.html) (modified to use [numba](http://numba.pydata.org/)).

<p float="left" align="middle">
  <img src="https://raw.githubusercontent.com/gauteh/fsinc/master/doc/example_1d.png" width="40%" />
  <img src="https://raw.githubusercontent.com/gauteh/fsinc/master/doc/example_2d.png" width="40%" />
</p>

Theory and details are described in more detail in [doc/fsinc.md](doc/fsinc.md) ([pdf](https://raw.githubusercontent.com/gauteh/fsinc/master/doc/fsinc.pdf)).

There are a couple of examples in [examples/](examples/), tests can be run with `pytest`. To show plots during testing use `pytest --plot`.

## Installation

```sh
pip install .
```

## Running local tests

From the `fsinc` top directory, make sure you have `pytest` and
`pytest-benchmark` installed in your python enviroment, then do:
```sh
$ pytest
```
To do a subset of tests, eg just 1d ones, use:
```sh
pytest -s -k sinc1d
```
Here the `-s` makes sure output is not caputred, while `-k` filters tests based
on the string.

> If you are having trouble with the `fastgl` module not being found, try using:
> `PYTHONPATH=fsinc pytest`.

## Building docs

Set up the environment in [doc/environment.yml](doc/environment.yml), install enough of tex-live and run `make` to generate `pdf` using `pandoc`.

# References

* A. H. Barnett, J. F. Magland, and L. af Klinteberg. A parallel non-uniform fast Fourier transform library based on an "exponential of semicircle" kernel. SIAM J. Sci. Comput. 41(5), C479-C504 (2019).

* Greengard, L., Lee, J. Y., & Inati, S. (2006). The fast sinc transform and image reconstruction from nonuniform samples in k-space. Communications in Applied Mathematics and Computational Science, 1(1), 121–131. [https://doi.org/10.2140/camcos.2006.1.121](https://doi.org/10.2140/camcos.2006.1.121)
