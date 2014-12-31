celp_codec
==========

This is a simple CELP voice codec implementation.
http://en.wikipedia.org/wiki/Code-excited_linear_prediction

Features
--------

* Encoding and decoding from multiple WAV and AIFF formats.
* Support file or stdin/stdout
* Real time processing, encode and decode from streams.
* Configurable
    - LPC order
    - Frame length
    - Number of subframes
    - Adaptive codebook size
    - Multiple frame windows ('hanning', 'boxcar', etc)
    - Weighting coefficients


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
    - Quantize LSP
    - Convert LSP to LPC
