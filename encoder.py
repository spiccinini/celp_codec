# -*- coding: utf8 -*-

import sys
import argparse

import numpy as np
import scipy

from ctsndfile.libsndfile import SndFile, OPEN_MODES, FILE_FORMATS

import celp

parser = argparse.ArgumentParser(description='CELP encoder.')
parser.add_argument('-i', type=str, default="-",
                    help='input file. Defaults to standard input')
parser.add_argument('-o', type=str, default="-",
                    help='output file. Defaults to standard output')

parser.add_argument('--lpc-order', type=int, default=10,
                    help='CELP LPC order')
parser.add_argument('--frame-length', type=int, default=160,
                    help='frame length')
parser.add_argument('--subframes', type=int, default=4,
                    help='numbre of subframes')
parser.add_argument('--fixed-cb-size', type=int, default=4,
                    help='fixed codebook size')

args = parser.parse_args()


in_file = SndFile(args.i)

assert in_file.samplerate == 8000
assert in_file.channels == 1

out_file = SndFile(args.o, open_mode=OPEN_MODES.SFM_WRITE,
                   writeSamplerate=8000,
                   writeFormat=FILE_FORMATS.SF_FORMAT_WAV|FILE_FORMATS.SF_FORMAT_PCM_16,
                   writeNbChannels=1)

#out_file = args.o

celp_encoder = celp.CELP(frame_length=args.frame_length,
                         n_subframes=args.subframes,
                         lpc_order=args.lpc_order,
                         fixed_codebook_size=args.fixed_cb_size)

celp_decoder = celp.CELP(frame_length=args.frame_length,
                         n_subframes=args.subframes,
                         lpc_order=args.lpc_order,
                         fixed_codebook_size=args.fixed_cb_size)

bytes_per_frame = celp_encoder.bytes_per_frame()

while True:
    frame, n_samples = in_file.read(args.frame_length, dtype="float64")

    if n_samples < args.frame_length:
        break
    bits = celp_encoder.encode(frame[:,0])

    frame = celp_decoder.decode(bits)
    out_file.write(frame)

out_file.close()

    #out_file.write(bits)
