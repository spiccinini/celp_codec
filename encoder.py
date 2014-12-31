# -*- coding: utf8 -*-

import sys
import argparse

import numpy as np
import scipy

np.seterr(all='raise')

from ctsndfile.libsndfile import SndFile, OPEN_MODES, FILE_FORMATS

import celp

parser = argparse.ArgumentParser(description='CELP encoder.')
parser.add_argument('-i', type=str, default="-",
                    help='input file. Defaults to standard input')
parser.add_argument('-o', type=argparse.FileType('w'), default=sys.stdout,
                    help='output file. Defaults to standard output')
parser.add_argument('--lpc-order', type=int, default=10,
                    help='CELP LPC order')
parser.add_argument('--frame-length', type=int, default=160,
                    help='frame length')
parser.add_argument('--subframes', type=int, default=4,
                    help='numbre of subframes')
parser.add_argument('--fixed-cb-size', type=int, default=128,
                    help='fixed codebook size')
parser.add_argument('--adapt-cb-size', type=int, default=80,
                    help='adaptive codebook size')
parser.add_argument('--frame-window', type=str, default="boxcar",
                    help='frame windows (boxcar, hamming, hanning, etc)')
parser.add_argument('--weigthing-coeff-1', type=float, default=0.9,
                    help='first coefficient of the weigthing filter'
                    ' W(z) = A(z/coeff_1) / A(z/coeff_2)')
parser.add_argument('--weigthing-coeff-2', type=float, default=0.6,
                    help='second coefficient of the weigthing filter'
                    ' W(z) = A(z/coeff_1) / A(z/coeff_2)')
args = parser.parse_args()


in_file = SndFile(args.i)

assert in_file.samplerate == 8000
assert in_file.channels == 1

out_file = args.o

celp_encoder = celp.CELP(frame_length=args.frame_length,
                         n_subframes=args.subframes,
                         lpc_order=args.lpc_order,
                         fixed_codebook_size=args.fixed_cb_size,
                         frame_window=args.frame_window,
                         weigthing_coeff_1=args.weigthing_coeff_1,
                         weigthing_coeff_2=args.weigthing_coeff_2)

bytes_per_frame = celp_encoder.bytes_per_frame()

while True:
    frame, n_samples = in_file.read(args.frame_length, dtype="float64")

    if n_samples < args.frame_length:
        break

    bits = celp_encoder.encode(frame[:,0])
    out_file.write(bits)

out_file.close()
