#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: demo.py
# Author: Irreq
# Date: 21/08-2022

import os
import time
import argparse

import numpy as np
from scipy.io import wavfile

from beamformer import create_antenna_array, get_filter_taps_direction, load_coefficients, beamform

import config

"""Usage:

If you want to listen to `RAW_FILE.npy` 0 degrees to the side and 45 degrees up:

python3 demo.py --path RAW_FILE.npy --play --save BEAMFORMED_FILE.wav 0 45

"""

def parse_args():
    parser = argparse.ArgumentParser(description="Time-domain beamformer")
    parser.add_argument("yaw", default=config.yaw, type=int, help="yaw.")
    parser.add_argument("pitch", type=int, help="pitch.")
    parser.add_argument(
        "--save",
        default=False,
        type=str,
        help="Save results to a wav file.",
    )
    parser.add_argument(
        "--play",
        default=False,
        action="store_true",
        help="Play results back.",
    )
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Save results to a wav file.",
    )
    parser.add_argument(
        "--rows",
        default=config.rows,
        type=int,
        help="Number of rows.",
    )
    parser.add_argument(
        "--columns",
        default=config.columns,
        type=int,
        help="Number of columns.",
    )
    parser.add_argument(
        "--distance",
        default=config.distance,
        type=int,
        help="Distance between the elements.",
    )
    parser.add_argument(
        "--propagation_speed",
        default=config.propagation_speed,
        type=int,
        help="propagation speed for signal.",
    )
    parser.add_argument(
        "--fs",
        default=config.fs,
        type=int,
        help="Sampling rate.",
    )
    parser.add_argument(
        "--position",
        default=config.position,
        type=list,
        help="Coordinate position of origin of array.",
    )
    parser.add_argument(
        "--path",
        # required=True,
        default=False,
        type=str,
        help="Path to .npy file.",
    )
    parser.add_argument(
        "--coefficients_path",
        default=config.coefficients_path,
        type=str,
        help="Path to coefficients file.",
    )

    return parser.parse_args()


def mimo(args, x=19, y=19, limit=180):

    filenames = next(os.walk("recorded/"), (None, None, []))[2]  # [] if no file
    signals = []
    for file_name in filenames:
        _, data = wavfile.read("recorded/"+file_name)
        signals.append(data)
    signals = np.asarray(signals)

    antenna_array = create_antenna_array(args.position, args.distance, args.rows, args.columns)
    
    coefficients = load_coefficients(args.coefficients_path)

    rows = np.linspace(-limit/3, limit/3, x)
    cols = np.linspace(-limit/3, limit/3, y)

    res = np.empty((x*y, 4))
    index = 0
    for i, row in enumerate(rows):
        for v, col in enumerate(cols):
            res[index] = [i, v, row, col]
            index+=1

    start = time.time()
    mimo = np.empty((x, y, 1))

    index = 0
    for i, v, yaw, pitch in res:
        taps = get_filter_taps_direction(yaw, pitch, antenna_array, args.fs, args.propagation_speed)
        beamformed = beamform(signals, taps, coefficients)
        power = np.sum(beamformed**2)/beamformed.size
        mimo[int(v)][18-int(i)] = power
        print(f"Processing: ["+int(index/(x*y)*50)*"="+int(50-index/(x*y)*50)*" "+"]", end="\r")
        index += 1

    print(f"Finished in {round(time.time()-start, 4)}s")

    mimo = mimo.reshape(x, y)
    mimo = np.flip(mimo, (1, 0))
    np.save(f"data/mimo.npy", mimo)

def miso(args):
    antenna_array = create_antenna_array(args.position, args.distance, args.rows, args.columns)
    taps = get_filter_taps_direction(args.yaw, args.pitch, antenna_array, args.fs, args.propagation_speed)
    coefficients = load_coefficients(args.coefficients_path)
    
    if not args.path:
    
        import os
        from scipy.io import wavfile

        filenames = next(os.walk("recorded/"), (None, None, []))[2]  # [] if no file
        signals = []
        for file_name in filenames:
            _, data = wavfile.read("recorded/"+file_name)
            signals.append(data)
        signals = np.asarray(signals)

    else:
        try:
            signals = np.load(args.path)
        except:
            print("Please enter a valid .npy file!")
            exit()

    beamformed = beamform(signals, taps, coefficients)
    if args.plot:
        import matplotlib.pyplot as plt
        plt.specgram(beamformed)
        plt.show()
    if args.save:
        from scipy.io import wavfile
        wavfile.write(args.save, args.fs, beamformed.astype(np.float32))

        if args.play:
            import os
            os.system("mpv "+args.save)

if __name__ == "__main__":
    args = parse_args()
    if args.play:
        miso(args)
    else:
        mimo(args)
