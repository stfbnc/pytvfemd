# pytvfemd.py
# Copyright (C) 2022-  Stefano Bianchi, Alessandro Longo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import pchip_interpolate
from scipy.signal import hilbert
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings

warnings.filterwarnings("ignore")


def tvfemd(x, thresh_bwr=0.1, bsp_order=26, modes=0, max_imf=None):
    """Time varying filter based EMD
    Parameters
    ----------
    x : numpy ndarray
        Input signal x(t).
    thresh_bwr : float, optional
        Instantaneous bandwidth threshold (default : 0.1).
    bsp_order : int, optional
        b-spline order (default : 26).
    modes : int, optional
        Extra imfs to extract, ignored if `max_imf` is not None (default : 0).
    max_imf : int, optional
        Maximum number of imfs to extract (default : None)
    Returns
    -------
    imf : numpy ndarray
        Imfs matrix.
    """
    if len(x.shape) != 1:
        raise ValueError("Input signal must be unidimensional.")

    if max_imf is None:
        num_max = int(np.floor(np.log2(len(x))) + 1)
        max_imf = num_max + modes
    else:
        max_imf = int(max_imf)
    if max_imf <= 0:
        raise ValueError("Parameter `max_imf` must be a positive integer.")

    imf = np.zeros((max_imf, len(x)), dtype=float)
    tmp_x = x.copy()

    nimf = 0
    for nimf in range(max_imf):
        indmin_x, indmax_x = __extr(tmp_x)

        if nimf == (max_imf - 1):
            imf[nimf, :] = tmp_x
            nimf += 1
            break

        if len(np.concatenate([indmin_x, indmax_x])) < 4:
            imf[nimf, :] = tmp_x
            if len(np.where(tmp_x != 0)[0]) > 0:
                nimf += 1
            break

        num_padding = int(np.round(len(tmp_x) * 0.5))
        y = tmp_x.copy()

        stop_iter = False
        ind_remov_pad = []
        for niter in range(100):
            y = np.concatenate([np.flip(y[1:2 + num_padding - 1]), y, np.flip(y[-num_padding - 1:-1])])

            ind_remov_pad = np.arange(num_padding, len(y) - num_padding, dtype=int)
            indmin_y, indmax_y = __extr(y)
            index_c_y = np.sort(np.concatenate([indmin_y, indmax_y]))
            inst_amp_0, inst_freq_0 = __inst_freq_local(y)

            # instantaneous amplitudes and frequencies, and bisecting frequency
            # LHF and LLF components
            a1, f1, a2, f2, bis_freq, inst_bwr, avg_freq = __divide_y(y, inst_amp_0, inst_freq_0)

            inst_bwr_2 = inst_bwr.copy()
            for j in range(0, len(index_c_y) - 2, 2):
                ind = np.arange(index_c_y[j], index_c_y[j + 2] + 1, dtype=int)
                inst_bwr_2[ind] = np.mean(inst_bwr[ind])

            bis_freq[inst_bwr_2 < thresh_bwr] = 1e-12
            bis_freq[bis_freq > 0.5] = 0.45
            bis_freq[bis_freq <= 0] = 1e-12

            bis_freq = __anti_modemixing(y, bis_freq, ind_remov_pad, num_padding)
            bis_freq = bis_freq[ind_remov_pad]
            bis_freq = np.concatenate([np.flip(bis_freq[1:2 + num_padding - 1]),
                                       bis_freq, np.flip(bis_freq[-num_padding - 1:-1])])

            bis_freq = __anti_modemixing(y, bis_freq, ind_remov_pad, num_padding)
            bis_freq = bis_freq[ind_remov_pad]
            bis_freq = np.concatenate([np.flip(bis_freq[1:2 + num_padding - 1]),
                                       bis_freq, np.flip(bis_freq[-num_padding - 1:-1])])

            temp_inst_bwr = inst_bwr_2[ind_remov_pad]
            ind_start = np.round(len(temp_inst_bwr) * 0.05).astype(int) - 1
            ind_end = np.round(len(temp_inst_bwr) * 0.95).astype(int) - 1

            if (((niter >= 1) and
                 (np.mean(temp_inst_bwr[ind_start:ind_end + 1]) < (thresh_bwr + thresh_bwr / 4 * (niter + 1)))) or
                    (niter >= 5) or
                    ((nimf > 0) and
                     (np.mean(temp_inst_bwr[ind_start:ind_end + 1]) < (thresh_bwr + thresh_bwr / 4 * (niter + 1))))):
                stop_iter = True
                break

            if (len(np.where(temp_inst_bwr[ind_start:ind_end + 1] > thresh_bwr)[0]) /
                len(inst_bwr_2[ind_remov_pad])) < 0.2:
                stop_iter = True
                break

            # integral of the bisecting frequency
            phi = np.zeros((len(bis_freq), ))
            for i in range(len(bis_freq) - 1):
                phi[i + 1] = phi[i] + 2 * np.pi * bis_freq[i]

            # knots as the extrema of h(t) = cos(phi)
            indmin_knot, indmax_knot = __extr(np.cos(phi))
            index_c_knot = np.sort(np.concatenate([indmin_knot, indmax_knot]))
            if len(index_c_knot) > 2:
                # obtaining LLF component
                localmean = __splinefit(np.arange(0, len(y), dtype=int), y, index_c_knot, bsp_order)
            else:
                stop_iter = True
                break

            if (np.max(np.abs(y[ind_remov_pad] - localmean[ind_remov_pad])) / np.min(
                    np.abs(localmean[ind_remov_pad])) < 1e-3):
                stop_iter = True
                break

            # sifting-like procedure, subtract LLF iteratively
            # until the LHF component is narrow band
            temp_residual = y - localmean
            temp_residual = temp_residual[ind_remov_pad]
            temp_residual = temp_residual[np.round(len(temp_residual) * 0.1).astype(int) - 1:
                                          -np.round(len(temp_residual) * 0.1).astype(int)]
            localmean2 = localmean[ind_remov_pad]
            localmean2 = localmean2[np.round(len(localmean2) * 0.1).astype(int) - 1:
                                    -np.round(len(localmean2) * 0.1).astype(int)]
            if (np.abs(np.max(localmean2)) / np.abs(np.max(inst_amp_0[ind_remov_pad])) < 3.5e-2 or
                    np.abs(np.max(temp_residual)) / np.abs(np.max(inst_amp_0[ind_remov_pad])) < 1e-2):
                stop_iter = True
                break

            y -= localmean
            y = y[ind_remov_pad]

        if stop_iter:
            imf[nimf, :] = y[ind_remov_pad]
            tmp_x -= y[ind_remov_pad]

    imf = np.delete(imf, np.s_[nimf:max_imf], axis=0)

    return imf.transpose((1, 0))


def __anti_modemixing(y, bis_freq, ind_remov_pad, num_padding):
    org_bis_freq = bis_freq.copy()
    flag_intermitt = 0
    t = np.arange(0, len(bis_freq), dtype=int)
    intermitt = np.array([], dtype=int)

    indmin_y, indmax_y = __extr(y)
    zero_span = np.array([], dtype=int)

    for i in range(1, len(indmax_y) - 1):
        time_span = np.arange(indmax_y[i - 1], indmax_y[i + 1] + 1, dtype=int)
        if (np.max(bis_freq[time_span]) - np.min(bis_freq[time_span])) / np.min(bis_freq[time_span]) > 0.25:
            zero_span = np.concatenate([zero_span, time_span])
    bis_freq[zero_span] = 0

    diff_bis_freq = np.zeros(bis_freq.shape)
    for i in range(len(indmax_y) - 1):
        time_span = np.arange(indmax_y[i], indmax_y[i + 1] + 1, dtype=int)
        if (np.max(bis_freq[time_span]) - np.min(bis_freq[time_span])) / np.min(bis_freq[time_span]) > 0.25:
            intermitt = np.concatenate([intermitt, [indmax_y[i]]])
            diff_bis_freq[indmax_y[i]] = bis_freq[indmax_y[i + 1]] - bis_freq[indmax_y[i]]

    ind_remov_pad = np.delete(ind_remov_pad,
                              np.r_[np.s_[0:np.round(0.1 * len(ind_remov_pad)).astype(int)],
                                    np.s_[np.round(0.9 * len(ind_remov_pad)).astype(int) - 1:len(ind_remov_pad)]])
    inters = np.intersect1d(ind_remov_pad, intermitt)
    if len(inters) > 0:
        flag_intermitt = 1

    for i in range(1, len(intermitt) - 1):
        u1 = intermitt[i - 1]
        u2 = intermitt[i]
        u3 = intermitt[i + 1]
        if diff_bis_freq[u2] > 0:
            bis_freq[u1:u2 + 1] = 0
        if diff_bis_freq[u2] < 0:
            bis_freq[u2:u3 + 1] = 0

    temp_bis_freq = bis_freq.copy()
    temp_bis_freq[temp_bis_freq < 1e-9] = 0
    temp_bis_freq = temp_bis_freq[ind_remov_pad]
    temp_bis_freq = np.concatenate([np.flip(temp_bis_freq[1:2 + num_padding - 1]), temp_bis_freq,
                                    np.flip(temp_bis_freq[-num_padding - 1:-1])])
    flip_bis_freq = np.flip(bis_freq)
    id_t = np.where(temp_bis_freq > 1e-9)[0]
    id_f = np.where(flip_bis_freq > 1e-9)[0]
    if len(id_t) > 0 and len(id_f) > 0:
        temp_bis_freq[0] = bis_freq[np.where(bis_freq > 1e-9)[0][0]]
        temp_bis_freq[-1] = flip_bis_freq[np.where(flip_bis_freq > 1e-9)[0][0]]
    else:
        temp_bis_freq[0] = bis_freq[0]
        temp_bis_freq[-1] = bis_freq[-1]

    bis_freq = temp_bis_freq.copy()
    if len(t[np.where(bis_freq != 0)[0]]) < 2:
        return

    bis_freq = pchip_interpolate(t[np.where(bis_freq != 0)[0]],
                                 bis_freq[np.where(bis_freq != 0)[0]], t)
    flip_bis_freq = np.flip(org_bis_freq)
    if len(np.where(org_bis_freq > 1e-9)[0]) > 0 and len(np.where(flip_bis_freq > 1e-9)[0]) > 0:
        org_bis_freq[0] = org_bis_freq[np.where(org_bis_freq > 1e-9)[0][0]]
        org_bis_freq[-1] = flip_bis_freq[np.where(flip_bis_freq > 1e-9)[0][0]]

    org_bis_freq[np.where(org_bis_freq < 1e-9)[0]] = 0
    org_bis_freq[0] = bis_freq[0]
    org_bis_freq[-1] = bis_freq[-1]
    org_bis_freq = pchip_interpolate(t[np.where(org_bis_freq != 0)[0]],
                                     org_bis_freq[np.where(org_bis_freq != 0)[0]], t)

    if flag_intermitt and np.max(temp_bis_freq[ind_remov_pad]) > 1e-9:
        output_cutoff = bis_freq.copy()
    else:
        output_cutoff = org_bis_freq.copy()

    output_cutoff[np.where(output_cutoff > 0.45)[0]] = 0.45
    output_cutoff[np.where(output_cutoff < 0)[0]] = 0

    return output_cutoff


def __divide_y(y, inst_amp_0, inst_freq_0):
    # divide y(t) into two sub-signals a1(t)exp(2*pi*f1(t)) and a2(t)exp(2*pi*f2(t)).
    # Parameters
    # ----------
    # y : numpy ndarray
    #     Input signal.
    # inst_amp_0 : numpy ndarray
    #     Instantaneous amplitude of `y`.
    # inst_freq_0 : numpy ndarray
    #     Instantaneous frequency of `y`.
    # Returns
    # -------
    # a1 : numpy ndarray
    #     Instantaneous amplitude of the first sub-signal.
    # f1 : numpy ndarray
    #     Instantaneous frequency of the first sub-signal.
    # a2 : numpy ndarray
    #     Instantaneous amplitude of the first sub-signal.
    # f2 : numpy ndarray
    #     Instantaneous frequency of the second sub-signal.
    # bis_freq : numpy ndarray
    #     Bisecting frequency, (`f1` + `f2`) / 2.
    # ratio_bw : numpy ndarray
    #     Instantaneous bandwidth ratio (stopping criterion).
    # avg_freq : numpy ndarray
    #     Average frequency.
    l_inst = len(inst_amp_0) if len(inst_amp_0.shape) == 1 else len(inst_amp_0[0])
    tt = np.arange(0, l_inst, dtype=int)
    squar_inst_amp_0 = np.power(inst_amp_0, 2.0)

    indmin_y, indmax_y = __extr(y)
    indmin_amp_0, indmax_amp_0 = __extr(squar_inst_amp_0)

    if len(indmin_amp_0) < 2 or len(indmax_amp_0) < 2:
        a1 = np.zeros(inst_amp_0.shape, dtype=float)
        a2 = np.zeros(inst_amp_0.shape, dtype=float)
        f1 = inst_freq_0.copy()
        f2 = inst_freq_0.copy()
        ratio_bw = a1.copy()
        bis_freq = np.zeros(inst_amp_0.shape, dtype=float)
        avg_freq = np.zeros(inst_amp_0.shape, dtype=float)

        return a1, f1, a2, f2, bis_freq, ratio_bw, avg_freq

    envpmax_inst_amp = pchip_interpolate(indmax_amp_0, inst_amp_0[indmax_amp_0], tt)
    envpmin_inst_amp = pchip_interpolate(indmin_amp_0, inst_amp_0[indmin_amp_0], tt)

    a1 = (envpmax_inst_amp + envpmin_inst_amp) / 2.0
    a2 = (envpmax_inst_amp - envpmin_inst_amp) / 2.0
    indmin_a2, indmax_a2 = __extr(a2)

    inst_amp_inst_amp_2 = inst_freq_0 * np.power(inst_amp_0, 2.0)
    inst_amp_tmax = pchip_interpolate(indmax_amp_0, inst_amp_inst_amp_2[indmax_amp_0],
                                      np.arange(0, len(inst_amp_inst_amp_2), dtype=int))
    inst_amp_tmin = pchip_interpolate(indmin_amp_0, inst_amp_inst_amp_2[indmin_amp_0],
                                      np.arange(0, len(inst_amp_inst_amp_2), dtype=int))
    f1 = np.zeros((len(inst_freq_0),), dtype=float)
    f2 = np.zeros((len(inst_freq_0),), dtype=float)
    for i in range(len(inst_freq_0)):
        a_mtx = np.empty((2, 2), dtype=float)
        a_mtx[0, :] = np.array([np.power(a1[i], 2.0) + a1[i] * a2[i], np.power(a2[i], 2.0) + a1[i] * a2[i]])
        a_mtx[1, :] = np.array([np.power(a1[i], 2.0) - a1[i] * a2[i], np.power(a2[i], 2.0) - a1[i] * a2[i]])
        b_mtx = np.array([inst_amp_tmax[i], inst_amp_tmin[i]])
        c_mtx = np.linalg.solve(a_mtx, b_mtx)
        f1[i] = c_mtx[0]
        f2[i] = c_mtx[1]

    bis_freq = (inst_amp_tmax - inst_amp_tmin) / (4 * a1 * a2)
    if len(indmax_a2) > 3:
        bis_freq = pchip_interpolate(indmax_a2, bis_freq[indmax_a2], tt)

    avg_freq = (inst_amp_tmax + inst_amp_tmin) / (2 * (np.power(a1, 2.0) + np.power(a2, 2.0)))
    cos_diffphi = (np.power(inst_amp_0, 2.0) - np.power(a1, 2.0) - np.power(a2, 2.0)) / (2 * a1 * a2)
    cos_diffphi[cos_diffphi > 1.2] = 1
    cos_diffphi[cos_diffphi < -1.2] = -1
    inst_amp1, inst_freq_diff_phi = __inst_freq_local(cos_diffphi)

    diff_a1 = (a1[2:] - a1[:-2]) / 2.0
    diff_a1 = np.concatenate([[diff_a1[0]], diff_a1, [diff_a1[-1]]])
    diff_a2 = (a2[2:] - a2[:-2]) / 2.0
    diff_a2 = np.concatenate([[diff_a2[0]], diff_a2, [diff_a2[-1]]])

    inst_bw = np.power(
        (np.power(diff_a1, 2.0) + np.power(diff_a2, 2.0)) / (np.power(a1, 2.0) + np.power(a2, 2.0)) + np.power(a1, 2.0) *
        np.power(a2, 2.0) * np.power(inst_freq_diff_phi, 2.0) / np.power(np.power(a1, 2.0) + np.power(a2, 2.0), 2.0), 0.5)
    ratio_bw = np.abs(inst_bw / avg_freq)
    ratio_bw[(a2 / a1) < 5e-3] = 0
    ratio_bw[avg_freq < 1e-7] = 0
    ratio_bw[ratio_bw > 1] = 1

    ff1 = (inst_freq_diff_phi + 2.0 * bis_freq) / 2.0
    ff2 = (2.0 * bis_freq - inst_freq_diff_phi) / 2.0
    f1[np.abs((a1 - a2) / a1) < 0.05] = ff1[np.abs((a1 - a2) / a1) < 0.05]
    f2[np.abs((a1 - a2) / a1) < 0.05] = ff2[np.abs((a1 - a2) / a1) < 0.05]

    temp_inst_amp_0 = inst_amp_0.copy()
    for j in range(len(indmax_y) - 1):
        ind = np.arange(indmax_y[j], indmax_y[j + 1] + 1, dtype=int)
        temp_inst_amp_0[ind] = np.mean(inst_amp_0[ind])

    ratio_bw[np.abs(temp_inst_amp_0) / np.max(np.abs(y)) < 5e-2] = 0
    f1[np.abs(temp_inst_amp_0) / np.max(np.abs(y)) < 4e-2] = 1 / len(y) / 1000
    f2[np.abs(temp_inst_amp_0) / np.max(np.abs(y)) < 4e-2] = 1 / len(y) / 1000
    bis_freq[bis_freq > 0.5] = 0.5
    bis_freq[bis_freq < 0] = 0

    return a1, f1, a2, f2, bis_freq, ratio_bw, avg_freq


def __extr(x):
    # Extracts the indices of extrema.
    # Parameters
    # ----------
    # x : numpy ndarray
    #     Input signal.
    # t : numpy ndarray, optional
    #     (default : range(len(x))).
    # Returns
    # -------
    # indmin : numpy ndarray
    #     Indices of minima.
    # indmax : numpy ndarray
    #     Indices of maxima.
    if len(x.shape) == 1:
        m = len(x)
        d = np.diff(x)
    elif x.shape[0] == 1:
        m = len(x[0])
        d = np.diff(x)[0]
    else:
        d = np.diff(x, axis=0)[0]
        m = len(x[0])

    d1 = d[:-1]
    d2 = d[1:]
    indmin = np.where((d1 * d2 < 0.0) & (d1 < 0.0))[0] + 1
    indmax = np.where((d1 * d2 < 0.0) & (d1 > 0.0))[0] + 1

    if len(np.where(d == 0.0)[0]) > 0:
        imax = np.array([], dtype=int)
        imin = np.array([], dtype=int)

        bad = (d == 0.0)
        c_bad = np.concatenate([[0], bad, [0]])
        dd = np.diff(c_bad)
        debs = np.where(dd == 1)[0]
        fins = np.where(dd == -1)[0]

        if len(debs) > 0 and debs[0] == 0:
            if len(debs) > 1:
                debs = debs[1:]
                fins = fins[1:]
            else:
                debs = np.array([], dtype=int)
                fins = np.array([], dtype=int)

        if len(debs) > 0:
            if fins[-1] == m - 1:
                if len(debs) > 1:
                    debs = debs[:-1]
                    fins = fins[:-1]
                else:
                    debs = np.array([], dtype=int)
                    fins = np.array([], dtype=int)

        if len(debs) > 0:
            for k in range(len(debs)):
                if d[debs[k] - 1] > 0:
                    if d[fins[k]] < 0:
                        imax = np.concatenate([imax, [np.round((fins[k] + debs[k]) / 2)]])
                else:
                    if d[fins[k]] > 0:
                        imin = np.concatenate([imin, [np.round((fins[k] + debs[k]) / 2)]])

        if len(imax) > 0:
            indmax = np.sort(np.concatenate([indmax, imax]))
        if len(imin) > 0:
            indmin = np.sort(np.concatenate([indmin, imin]))

    return indmin.astype(int), indmax.astype(int)


def __inst_freq_local(data):
    # Computes Hilbert-Huang spectrum using Hilbert transform,
    # the instantaneous frequency and the instantaneous amplitude for each IMF.
    # Parameters
    # ----------
    # data : numpy ndarray
    #     IMF matrix
    # Returns
    # -------
    # inst_amp : numpy ndarray
    #     Instantaneous amplitude matrix
    # inst_freq : numpy ndarray
    #     Instantaneous frequency matrix
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
        inst_freq_temp = (phi[2:] - phi[:-2]) / 2
        inst_freq_temp = np.concatenate([[inst_freq_temp[0]], inst_freq_temp, [inst_freq_temp[-1]]])
        inst_freq[k, :] = inst_freq_temp.flatten() / (2 * np.pi)

    inst_amp[0, 0] = inst_amp[0, 1]
    inst_amp[0, -1] = inst_amp[0, -2]

    inst_freq[np.where(inst_freq <= 0.0)] = 0.0

    return inst_amp[0], inst_freq[0]


def __splinefit(x, y, breaks, n):
    # Fit a spline to data.
    # Parameters
    # ----------
    # x : numpy ndarray
    #     x values.
    # y : numpy ndarray
    #     y values.
    # breaks : numpy ndarray
    #     Knots.
    # n : int
    #     Spline order.
    # Returns
    # -------
    # sp_fit : numpy ndarray
    #     Spline fit values.
    x, y, breaks = __check_knots(x, y, breaks)
    pp_dict = __splinebase(breaks, n)
    pieces = pp_dict["pieces"]
    a_sp = __spline_eval(pp_dict, x)
    ibin = np.digitize(x, breaks[1:-1])

    mx = len(x)
    ii = np.vstack((ibin, np.ones((n - 1, mx)))).astype(int)
    ii = np.cumsum(ii, axis=0)
    jj = np.tile(np.arange(0, mx).astype(int), (n, 1))
    ii = np.mod(ii, pieces)
    a_sp = csr_matrix((a_sp.flatten(), (ii.flatten(), jj.flatten())),
                      shape=(pieces, mx), dtype=float)
    a_sp.eliminate_zeros()

    if pieces < 20 * n / np.log(1.7 * n):
        a_sp = a_sp.todense().transpose()
        u = np.linalg.lstsq(a_sp, y)[0]
    else:
        u = spsolve(a_sp * a_sp.T, a_sp * csr_matrix(y, dtype=float).T)

    jj = np.mod(np.arange(0, pieces + n - 1, dtype=int), pieces)
    u = u[jj]

    ii = np.vstack((np.tile(np.arange(0, pieces, dtype=int), n), np.ones((n - 1, n * pieces))))
    ii = np.cumsum(ii, axis=0)
    jj = np.tile(np.arange(0, n * pieces, dtype=int), (n, 1))
    c_mtx = csr_matrix((pp_dict["coefs"].flatten("F"), (ii.flatten("F"), jj.flatten("F"))),
                       shape=(pieces + n - 1, n * pieces), dtype=float)
    coefs = u * c_mtx
    coefs = np.reshape(coefs, (int(len(coefs) / n), n), order="F")

    pp_spline = __pp_struct(breaks, coefs, 1)
    sp_fit = __spline_eval(pp_spline, np.arange(0, len(y), dtype=int))

    return sp_fit[0]


def __check_knots(x, y, knots):
    # Check if x points are outside knots range.
    # Parameters
    # ----------
    # x : numpy ndarray
    #     x values.
    # y : numpy ndarray
    #     y values.
    # knots : numpy ndarray
    #     Knots.
    # Returns
    # -------
    # x : numpy ndarray
    #     x values all in the knots range.
    # y : numpy ndarray
    #     y values all in the knots range.
    # knots : numpy ndarray
    #     Unique knots.
    if len(np.where(np.diff(knots) <= 0)[0]) > 0:
        knots = np.unique(knots)

    h = np.diff(knots)
    xlim1 = knots[0] - 0.01 * h[0]
    xlim2 = knots[-1] + 0.01 * h[-1]
    if x[0] < xlim1 or x[-1] > xlim2:
        p = knots[-1] - knots[0]
        x = ((x - knots[0]) % p) + knots[0]
        isort = np.argsort(x, kind="stable")
        x = x[isort]
        y = y[isort]

    return x, y, knots


def __splinebase(breaks, n):
    # Generates B-spline base of order `n` for knots `breaks`.
    # Parameters
    # ----------
    # breaks : numpy ndarray
    #     Knots.
    # n : int
    #     Spline order.
    # Returns
    # -------
    # pp : numpy ndarray
    #     B-spline base.
    breaks = breaks.flatten()
    breaks0 = breaks.copy()
    h = np.diff(breaks)
    pieces = len(h)
    deg = n - 1

    if deg > 0:
        if deg <= pieces:
            hcopy = h.copy()
        else:
            hcopy = np.tile(h, (int(np.ceil(deg / pieces)), ))
        hl = hcopy[-1:-deg - 1:-1]
        bl = breaks[0] - np.cumsum(hl)
        hr = hcopy[:deg]
        br = breaks[-1] + np.cumsum(hr)
        breaks = np.concatenate([bl[deg - 1::-1], breaks, br])
        h = np.diff(breaks)
        pieces = len(h)

    coefs = np.zeros((n * pieces, n), dtype=float)
    coefs[::n, 0] = 1

    ii = np.ones((deg + 1, pieces), dtype=int)
    ii[0, :] = np.linspace(0, pieces, pieces, endpoint=False, dtype=int)
    ii = np.cumsum(ii, axis=0)
    ii[np.where(ii > pieces - 1)] = pieces - 1
    hh = h[ii.flatten("F")]

    for k in range(1, n):
        for j in range(k):
            coefs[:, j] = coefs[:, j] * hh / (k - j)
        q = np.sum(coefs, axis=1)
        q = q.reshape((pieces, n)).T
        q = np.cumsum(q, axis=0)
        c0 = np.concatenate([np.zeros((1, pieces)), q[0:deg, :]])
        coefs[:, k] = c0.flatten("F")
        fmax = np.tile(q[n-1, :], (n, 1))
        fmax = fmax.flatten("F")
        for j in range(k + 1):
            coefs[:, j] = coefs[:, j] / fmax
        coefs[0:-deg, 0:k + 1] = coefs[0:-deg, 0:k + 1] - coefs[n - 1:, 0:k + 1]
        coefs[::n, k] = 0

    scale = np.ones(hh.shape)
    for k in range(n - 1):
        scale = scale / hh
        coefs[:, n - k - 2] = scale * coefs[:, n - k - 2]

    pieces -= 2 * deg

    ii = np.ones((deg + 1, pieces), dtype=int) * deg
    ii[0, :] = n * np.arange(1, pieces + 1, dtype=int)
    ii = np.cumsum(ii, axis=0) - 1
    coefs = coefs[ii.flatten("F"), :]

    return __pp_struct(breaks0, coefs, n)


def __pp_struct(br, cf, d):
    # Structure for piecewise polynomial parameters.
    # Parameters
    # ----------
    # br : numpy ndarray
    #     Knots.
    # cf : numpy ndarray
    #     Polynomial coefficients.
    # d : int
    #     Polynomials order.
    # Returns
    # -------
    # pp : dictionary
    #     Piecewise polynomial parameters structure.
    dlk = cf.shape[0] * cf.shape[1]
    l = len(br) - 1
    dl = d * l
    k = np.fix(dlk / dl + 100 * np.spacing(1)).astype(int)

    pp = {"breaks": br.reshape((1, l + 1))[0],
          "coefs": cf.reshape((dl, k)),
          "pieces": l,
          "order": k,
          "dim": d}

    return pp


def __spline_eval(pp, xx):
    # Evaluates piecewise polynomial.
    # Parameters
    # ----------
    # pp : dictionary
    #     Splines base parameters.
    # xx : numpy ndarray
    #     Evaluation points.
    # Returns
    # -------
    # v : numpy ndarray
    #     Piecewise polynomial values.
    sizexx = xx.shape
    lx = np.prod([s for s in xx.shape]).astype(int)
    xs = xx.reshape((lx, ))
    if len(sizexx) == 2 and sizexx[0] == 1:
        sizexx = (sizexx[1], )

    b = pp["breaks"]
    c = pp["coefs"]
    l = pp["pieces"]
    k = pp["order"]
    dd = pp["dim"]

    if lx > 0:
        index = np.digitize(xs, b[1:l])
    else:
        index = np.ones((lx, ), dtype=int)

    infxs = np.where(xs == np.inf)[0]
    if len(infxs) != 0:
        index[infxs] = l
    nogoodxs = np.where(index < 0)[0]
    if len(nogoodxs) != 0:
        xs[nogoodxs] = -999
        index[nogoodxs] = 1

    xs = xs - b[index]
    d = np.prod(dd)

    if d > 1:
        xs = np.tile(xs, (d, 1)).transpose((1, 0)).reshape((d * lx, ))
        index = d * (index + 1) - 1
        temp = np.arange(-d, 0).astype(int)
        arr = np.tile(temp[np.newaxis].transpose(), (1, lx)) + np.tile(index, (d, 1)) + 1
        index = arr.transpose((1, 0)).reshape((d * lx, ))
    else:
        if len(sizexx) > 1:
            dd = np.array([])
        else:
            dd = 1

    v = c[index, 0]
    for i in range(1, k):
        v = xs * v + c[index, i]

    if len(nogoodxs) > 0 and k == 1 and l > 1:
        v = v.reshape((d, lx))
        v[:, nogoodxs] = -999
    v = np.reshape(v, (dd, sizexx[0]), order="F")

    return v
