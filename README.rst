celp_codec
==========

This is a simple CELP implementation in python.

Dependencies
------------

* numpy
* scipy
* ctlibsnd (ṕython libsndfile wrapper) https://bitbucket.org/san/ctlibsnd

Future improvements
-------------------

* Implement fixed codebooks of different sizes
* Support more sample rates (currently only 8 kHz)
* Preprocesing filter (high pass, and equalization)
* Convert LPC (linear prediction coefs) to LSP (linear spectral pairs)
    * Quantize LSP
    * Convert LSP to LPC
