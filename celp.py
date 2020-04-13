# -*- coding: utf8 -*-
#    Copyright (C) 2011, 2014 Santiago Piccinini <piccinini santiago at gmail dot com>

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import scipy
import scipy.signal
from scipy.signal import lfilter
import scipy.linalg
from itertools import islice

import lpc
from adaptive_codebook import AdaptiveCodebook
from fixed_codebook_128x40 import FC

SAMPLE_RATE = 8000

class CELP(object):

    lpc_error_coefs_dtype = np.dtype(np.float16)
    amplifs_dtype = np.dtype(np.float16)
    index_dtype = np.dtype(np.uint8)

    def __init__(self, frame_length=160, n_subframes=4, lpc_order=10,
                 fixed_codebook_size=128, adaptive_codebook_size=160,
                 frame_window="boxcar", weigthing_coeff_1=0.9,
                 weigthing_coeff_2=0.6):

        self.frame_length = frame_length
        self.n_subframes = n_subframes
        self.lpc_order = lpc_order
        self.fc_size = fixed_codebook_size
        self.ac_size = adaptive_codebook_size

        self.subframe_length = self.frame_length / self.n_subframes
        self.zero_input = np.zeros(self.frame_length)
        self.delta = np.concatenate(([1.], np.zeros(self.subframe_length - 1)))

        self.frame_window = scipy.signal.get_window(frame_window, frame_length)

        self.weigthing_coeff_1, self.weigthing_coeff_2 = weigthing_coeff_1, weigthing_coeff_2

        self.fixed_codebook = FC

        self.adaptive_codebook = AdaptiveCodebook(vector_size=self.subframe_length,
                                                  cb_size=adaptive_codebook_size,
                                                  min_period=20)

        self._excitation = np.zeros(self.subframe_length)

    def encode(self, frame):
        import lpc
        # Apply window
        frame *= self.frame_window

        # Generate LPC coefficients
        lpc_error_coeffs = lpc.lpc_ref(frame, self.lpc_order)

        out_fc_indexes, out_ac_indexes = [], []
        out_fc_amplifs, out_ac_amplifs = [], []

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

            # Search the best adaptive signal
            M = np.dot(W, H)
            d = np.dot(W, subframe - z0) + z1
            ac_index, ac_amplif = search_codebook(M, d, self.adaptive_codebook)
            ac_excitation = self.adaptive_codebook[ac_index] * ac_amplif

            # Search the best fixed codebook signal
            d = np.dot(W, subframe - z0 - np.dot(H, ac_excitation) + z1)
            fc_index, fc_amplif = search_codebook(M, d, self.fixed_codebook)
            fc_excitation = fc_amplif * self.fixed_codebook[fc_index]

            # Build current excitation using adaptive and fixed codebooks
            self._excitation = ac_excitation + fc_excitation

            # Append current excitation to adaptive codebook
            self.adaptive_codebook.add_vector(self._excitation)

            # store output values for this frame
            out_fc_indexes.append(fc_index)
            out_fc_amplifs.append(fc_amplif)
            out_ac_indexes.append(ac_index)
            out_ac_amplifs.append(ac_amplif)

        # generate binary output string with all output values
        lpc = lpc_error_coeffs.astype(self.lpc_error_coefs_dtype).tostring()
        fc_indexes = np.array(out_fc_indexes, dtype=self.index_dtype).tostring()
        fc_amplifs = np.array(out_fc_amplifs, dtype=self.amplifs_dtype).tostring()
        ac_indexes = np.array(out_ac_indexes, dtype=self.index_dtype).tostring()
        ac_amplifs = np.array(out_ac_amplifs, dtype=self.amplifs_dtype).tostring()

        return lpc + fc_indexes + fc_amplifs + ac_indexes + ac_amplifs


    def bytes_per_frame(self):
        size = self.size_of_lpc() + self.size_of_indexes() * 2 + self.size_of_amplifs() * 2
        return size

    def size_of_lpc(self):
        return self.lpc_error_coefs_dtype.itemsize * (self.lpc_order + 1)

    def size_of_indexes(self):
        return self.index_dtype.itemsize * self.n_subframes

    def size_of_amplifs(self):
        return self.amplifs_dtype.itemsize * self.n_subframes

    def decode(self, frame_bits):
        frame_bits_it = iter(frame_bits)


        # read input values from input bits
        lpc_error_coeffs = np.fromstring("".join(islice(frame_bits_it, self.size_of_lpc())),
                                         dtype=self.lpc_error_coefs_dtype)
        fc_indexes = np.fromstring("".join(islice(frame_bits_it, self.size_of_indexes())),
                                   dtype=self.index_dtype)
        fc_amplifs = np.fromstring("".join(islice(frame_bits_it, self.size_of_amplifs())),
                                   dtype=self.amplifs_dtype)
        ac_indexes = np.fromstring("".join(islice(frame_bits_it, self.size_of_indexes())),
                                   dtype=self.index_dtype)
        ac_amplifs = np.fromstring("".join(islice(frame_bits_it, self.size_of_amplifs())),
                                   dtype=self.amplifs_dtype)
        out = np.array([])

        h = lfilter([1], lpc_error_coeffs, self.delta)
        H = scipy.linalg.toeplitz(h, np.concatenate(([h[0]], self.zero_input[:self.subframe_length-1])))

        for fc_index, fc_amplif, ac_index, ac_amplif in zip(fc_indexes, fc_amplifs, ac_indexes, ac_amplifs):
            z0 = lfilter([1], lpc_error_coeffs,
                         np.concatenate((self._excitation, np.zeros(self.subframe_length))))[self.subframe_length:]
            self._excitation = fc_amplif * self.fixed_codebook[fc_index] + ac_amplif * self.adaptive_codebook[ac_index]

            subframe_out = np.dot(H, self._excitation) + z0
            out = np.concatenate((out, subframe_out))

            # Append current excitation to adaptive codebook
            self.adaptive_codebook.add_vector(self._excitation)

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
        if not denominator:
            denominator = 1
        val = numerator / denominator
        if val > maximum:
            max_index = index
            maximum = val

    best_codevector = codebook[max_index]

    #       d^T M v
    # a = -----------   With v the best codevector
    #     v^T M^T M v
    denominator = np.dot(np.dot(best_codevector.T, M2), best_codevector)
    if not denominator:
        denominator = 1
    best_amplification = np.dot(dM, best_codevector) / denominator

    return max_index, best_amplification
