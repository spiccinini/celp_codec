# -*- coding: utf8 -*-

import numpy as np
import scipy
import scipy.signal
from scipy.signal import lfilter
import scipy.linalg
import py_lpc
import scipy.io.wavfile
import pylab

SAMPLE_RATE = 8000
FRAME_LENGTH = 160 # 0.02s @8000Hz
N_SUBFRAMES = 4
SUBFRAME_LENGTH =  FRAME_LENGTH / N_SUBFRAMES

LPC_ORDER = 10

FC_SIZE = 128

ZERO_INPUT = np.zeros(FRAME_LENGTH)

DELTA = np.concatenate(([1.], np.zeros(SUBFRAME_LENGTH-1)))

lpc_init_cond = np.zeros(LPC_ORDER)

# TODO:
#
# * Build LPC
# * Preprocesing filter (high pass, and equalization)
# * Build A(z), the prediction filter, from LPC.
# * Build W(z), the noise weighting filter, from A(z).
# * Convert LPC (linear prediction coefs) to LSP (linear spectral pairs)
# * Quantize LSP
# * Convert LSP to LPC
# * Build window for frame
# * Search in codebook
# * Adaptive Codebook (AC)

# CELP Simplest Algorithm (without preprocesing, AC, W(z), and quantization):
#
# For each frame:
# * Generate LPC from samples
# * Build H (convolution matrix) from H(z) = 1 / A(z)
# * For each subframe:
#   * Search in FC


def gen_gaussian_fixed_codebook(size):
    return np.random.normal(0, 1, size)

#fixed_codebook = gen_gaussian_fixed_codebook((FC_SIZE, SUBFRAME_LENGTH))
from fixed_codebook_128x40 import FC
fixed_codebook = FC

def search_codebook(M, d, codebook):
    max_index = 0
    maximum = 0

    dM = np.dot(d.T, M)
    M2 = np.dot(M.T, M)
    for index, codevector in enumerate(codebook):
        # (d^T M v)^2
        # -----------  Maximize this
        # v^T M^T M v
        numerator = np.dot(dM, codevector)**2
        denominator = np.dot(np.dot(codevector.T, M2), codevector)
        val = numerator / denominator
        if val > maximum:
            max_index = index
            maximum = val

    best_codevector = codebook[max_index]

    #       d^T M v
    # a = -----------   With v the best codevector
    #     v^T M^T M v
    best_amplification = np.dot(dM, best_codevector) / np.dot(np.dot(best_codevector.T, M2),
                                                              best_codevector)

    return max_index, best_amplification

SR, signal = scipy.io.wavfile.read('./data/mike_8.wav')
signal = signal.astype("float64")
signal /= np.max(signal)
assert(SR==8000)


signal = np.concatenate((signal, np.zeros(FRAME_LENGTH - (len(signal) % FRAME_LENGTH))))
signal = signal.reshape((len(signal)/FRAME_LENGTH, FRAME_LENGTH))

frames = signal

last_excitation_code = np.zeros(SUBFRAME_LENGTH)

out_amplifs = []
out_lpc_coefs = []
out_fc_index = []

# Encoding

for frame in frames:
    #import ipdb;ipdb.set_trace()
    #print "Frame codification"
    lpc_error_coeffs = py_lpc.lpc_ref(frame, LPC_ORDER)
    out_lpc_coefs.append(lpc_error_coeffs)
    # Buid the H matrix
    h = lfilter([1], lpc_error_coeffs, DELTA)
    H = scipy.linalg.toeplitz(h, np.concatenate(([h[0]], ZERO_INPUT[:SUBFRAME_LENGTH-1])))
    #import ipdb;ipdb.set_trace()

    #print "LPC coeffs %s"  % lpc_error_coeffs

    for subframe in frame.reshape((N_SUBFRAMES, SUBFRAME_LENGTH)):
        #print "\tSubframe codification"
        #lpc_zero_input_response, _ =  scipy.signal.lfilter([1], lpc_error_coeffs,
        #                                               ZERO_INPUT, zi=lpc_init_cond)
        #print lpc_error_coeffs
        z0 = lfilter([1], lpc_error_coeffs,
                     np.concatenate((last_excitation_code, np.zeros(SUBFRAME_LENGTH))))[SUBFRAME_LENGTH:]
        M = H
        d = subframe - z0
        index, amplif = search_codebook(M, d, fixed_codebook)

        last_excitation_code = amplif * fixed_codebook[index]
        #import ipdb;ipdb.set_trace()
        #pylab.plot(subframe)
        #pylab.plot(np.dot(H, last_excitation_code)+z0)

        print "\tindex: %d, amplif: %f" % (index, amplif)
        out_amplifs.append(amplif)
        out_fc_index.append(index)

print out_amplifs
print out_fc_index
print out_lpc_coefs

# Decoding

out_signal = np.array([])
excitation = np.zeros(SUBFRAME_LENGTH)

amplifs, fc_indexs = iter(out_amplifs), iter(out_fc_index)

for lpc_error_coeffs in out_lpc_coefs:

    h = lfilter([1], lpc_error_coeffs, DELTA)
    H = scipy.linalg.toeplitz(h, np.concatenate(([h[0]], ZERO_INPUT[:SUBFRAME_LENGTH-1])))

    for subframe in range(4):
        z0 = lfilter([1], lpc_error_coeffs,
                     np.concatenate((excitation, np.zeros(SUBFRAME_LENGTH))))[SUBFRAME_LENGTH:]

        amplif, index  = amplifs.next(), fc_indexs.next()
        excitation = amplif * fixed_codebook[index]

        out = np.dot(H, excitation) + z0

        out_signal = np.concatenate((out_signal, out))

out_signal = out_signal * np.iinfo(np.int16).max
scipy.io.wavfile.write('./data/mike_8_out.wav', 8000, out_signal.astype("int16"))


