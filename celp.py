# -*- coding: utf8 -*-

import numpy as np
import scipy
import scipy.signal
from scipy.signal import lfilter
import scipy.linalg
from itertools import islice

import lpc

SAMPLE_RATE = 8000

# TODO:
#
# * Preprocesing filter (high pass, and equalization)
# * Build W(z), the noise weighting filter, from A(z).
# * Convert LPC (linear prediction coefs) to LSP (linear spectral pairs)
# * Quantize LSP
# * Convert LSP to LPC
# * Build window for frame
# * Adaptive Codebook (AC)

# CELP Simplest Algorithm (without preprocesing, AC, W(z), and quantization):
#
# For each frame:
# * Generate LPC from samples
# * Build H (convolution matrix) from H(z) = 1 / A(z)
# * For each subframe:
#   * Search in FC

class CELP(object):

    lpc_error_coefs_dtype = np.dtype(np.float32)
    amplifs_dtype = np.dtype(np.float32)
    index_dtype = np.dtype(np.int16)

    def __init__(self, frame_length=160, n_subframes=4, lpc_order=10,
                 fixed_codebook_size=128, frame_window="boxcar",
                 weigthing_coeff_1=0.9, weigthing_coeff_2=0.6):

        self.frame_length = frame_length
        self.n_subframes = n_subframes
        self.lpc_order = lpc_order
        self.fc_size = fixed_codebook_size

        self.subframe_length = self.frame_length / self.n_subframes
        self.zero_input = np.zeros(self.frame_length)
        self.delta = np.concatenate(([1.], np.zeros(self.subframe_length - 1)))

        self.frame_window = scipy.signal.get_window(frame_window, frame_length)

        self.weigthing_coeff_1, self.weigthing_coeff_2 = weigthing_coeff_1, weigthing_coeff_2

        # TODO
        from fixed_codebook_128x40 import FC
        self.fixed_codebook = FC

        self._excitation = np.zeros(self.subframe_length)

    def encode(self, frame):
        import lpc
        frame *= self.frame_window
        lpc_error_coeffs = lpc.lpc_ref(frame, self.lpc_order)

        out_fc_indexes = []
        out_fc_amplifs = []

        # Buld the noise weigthing filter W matrix
        # W(z) = A(z/weigthing_coeff_1) / A(z/weigthing_coeff_2)
        weigthing_b = lpc_error_coeffs * np.power(self.weigthing_coeff_1, np.arange(self.lpc_order+1))
        weigthing_a = lpc_error_coeffs * np.power(self.weigthing_coeff_2, np.arange(self.lpc_order+1))
        w = lfilter(weigthing_b, weigthing_a, self.delta)
        W = scipy.linalg.toeplitz(w, np.concatenate(([w[0]], self.zero_input[:self.subframe_length-1])))

        # Buid the H matrix = 1 / A
        h = lfilter([1], lpc_error_coeffs, self.delta)
        H = scipy.linalg.toeplitz(h, np.concatenate(([h[0]], self.zero_input[:self.subframe_length-1])))

        for subframe in frame.reshape((self.n_subframes, self.subframe_length)):
            lpc_filtered = lfilter([1], lpc_error_coeffs,
                                   np.concatenate((self._excitation, np.zeros(self.subframe_length))))

            z0 = lpc_filtered[self.subframe_length:] # Zero input response for the H filter
            z1 = lfilter(weigthing_b, weigthing_a,   # Zero input response for the W filter
                         np.concatenate((lpc_filtered[:self.subframe_length],
                                        np.zeros(self.subframe_length))))[self.subframe_length:]

            M = np.dot(W, H)
            d = np.dot(W, subframe - z0) + z1
            fc_index, fc_amplif = search_codebook(M, d, self.fixed_codebook)

            self._excitation = fc_amplif * self.fixed_codebook[fc_index]

            out_fc_indexes.append(fc_index)
            out_fc_amplifs.append(fc_amplif)

        lpc = lpc_error_coeffs.astype(self.lpc_error_coefs_dtype).tostring()
        fc_indexes = np.array(out_fc_indexes, dtype=self.index_dtype).tostring()
        fc_amplifs = np.array(out_fc_amplifs, dtype=self.amplifs_dtype).tostring()

        return lpc + fc_indexes + fc_amplifs


    def bytes_per_frame(self):
        size = self.size_of_lpc() + self.size_of_fc_indexes() + self.size_of_amplifs()
        return size

    def size_of_lpc(self):
        return self.lpc_error_coefs_dtype.itemsize * (self.lpc_order + 1)

    def size_of_fc_indexes(self):
        return self.index_dtype.itemsize * self.n_subframes

    def size_of_amplifs(self):
        return self.amplifs_dtype.itemsize * self.n_subframes

    def decode(self, bits):
        it = iter(bits)

        lpc_error_coeffs = np.fromstring("".join(islice(it, self.size_of_lpc())),
                                         dtype=self.lpc_error_coefs_dtype)
        fc_indexes = np.fromstring("".join(islice(it, self.size_of_fc_indexes())),
                                   dtype=self.index_dtype)
        fc_amplifs = np.fromstring("".join(islice(it, self.size_of_amplifs())),
                                   dtype=self.amplifs_dtype)
        out = np.array([])

        h = lfilter([1], lpc_error_coeffs, self.delta)
        H = scipy.linalg.toeplitz(h, np.concatenate(([h[0]], self.zero_input[:self.subframe_length-1])))

        for fc_index, fc_amplif in zip(fc_indexes, fc_amplifs):
            z0 = lfilter([1], lpc_error_coeffs,
                         np.concatenate((self._excitation, np.zeros(self.subframe_length))))[self.subframe_length:]

            self._excitation = fc_amplif * self.fixed_codebook[fc_index]

            subframe_out = np.dot(H, self._excitation) + z0

            out = np.concatenate((out, subframe_out))

        return out


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
