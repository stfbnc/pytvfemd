### inst_freq_local.py
### Copyright (C) 2020  Stefano Bianchi
###
### This program is free software: you can redistribute it and/or modify
### it under the terms of the GNU General Public License as published by
### the Free Software Foundation, either version 3 of the License, or
### (at your option) any later version.
###
### This program is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
### GNU General Public License for more details.
###
### You should have received a copy of the GNU General Public License
### along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.signal import hilbert


def inst_freq_local(data):
    """Computes Hilbert-Huang spectrum using Hilbert transform,
    the instantaneous frequency and the instantaneous amplitude for each IMF.
    Parameters
    ----------
    data : numpy ndarray
        IMF matrix
    Returns
    -------
    inst_amp : numpy ndarray
        Instantaneous amplitude matrix
    inst_freq : numpy ndarray
        Instantaneous frequency matrix
    """
    fs = 1
    ts = 1 / fs

    dimension = data.shape
    if len(dimension) == 1:
        dimension = (1, dimension[0])
        data = np.expand_dims(data, axis=0)
    inst_amp = np.zeros(dimension, dtype=float)
    inst_freq = np.zeros(dimension, dtype=float)

    for k in range(dimension[0]):
        h = hilbert(data[k, :])

        inst_amp_temp = np.abs(h)
        inst_amp[k, :] = inst_amp_temp.flatten()
    
        phi = np.unwrap(np.angle(h))
        inst_freq_temp = (phi[2:] - phi[:-2]) / (2 * ts)
        inst_freq_temp = np.concatenate([[inst_freq_temp[0]], inst_freq_temp, [inst_freq_temp[-1]]])
        inst_freq[k, :] = inst_freq_temp.flatten() / (2 * np.pi)

    inst_amp[0, 0] = inst_amp[0, 1]
    inst_amp[0, -1] = inst_amp[0, -2]

    inst_freq[np.where(inst_freq <= 0.0)] = 0.0
    inst_freq = inst_freq / fs

    return inst_amp[0], inst_freq[0]



