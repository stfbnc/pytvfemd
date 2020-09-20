### splinefit.py
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def splinefit(x, y, breaks, n):
    """Fit a spline to data.
    Parameters
    ----------
    x : numpy ndarray
        x values.
    y : numpy ndarray
        y values.
    breaks : numpy ndarray
        Knots.
    n : int
        Spline order.
    Returns
    -------
    sp_fit : numpy ndarray
        Spline fit values.
    """
    x, y, breaks = check_knots(x, y, breaks)
    pp_dict = splinebase(breaks, n)
    pieces = pp_dict["pieces"]
    A = spline_eval(pp_dict, x)
    ibin = np.digitize(x, breaks[1:-1])

    mx = len(x)
    ii = np.vstack((ibin, np.ones((n - 1, mx)))).astype(int)
    ii = np.cumsum(ii, axis=0)
    jj = np.tile(np.arange(0, mx).astype(int), (n, 1))
    ii = np.mod(ii, pieces)
    A = csr_matrix((A.flatten(), (ii.flatten(), jj.flatten())),
                   shape=(pieces, mx), dtype=np.float)
    A.eliminate_zeros()

    if pieces < 20 * n / np.log(1.7 * n):
        A = A.todense().transpose()
        u = np.linalg.lstsq(A, y)[0]
    else:
        u = spsolve(A * A.T, A * csr_matrix(y, dtype=float).T)

    jj = np.mod(np.arange(0, pieces + n - 1, dtype=int), pieces)
    u = u[jj]

    ii = np.vstack((np.tile(np.arange(0, pieces, dtype=int), n), np.ones((n - 1, n * pieces))))
    ii = np.cumsum(ii, axis=0)
    jj = np.tile(np.arange(0, n * pieces, dtype=int), (n, 1))
    C = csr_matrix((pp_dict["coefs"].flatten("F"), (ii.flatten("F"), jj.flatten("F"))),
                   shape=(pieces + n - 1, n * pieces), dtype=float)
    coefs = u * C
    coefs = np.reshape(coefs, (int(len(coefs) / n), n), order="F")

    pp_spline = pp_struct(breaks, coefs, 1)
    sp_fit = spline_eval(pp_spline, np.arange(0, len(y), dtype=int))

    return sp_fit[0]


def check_knots(x, y, knots):
    """Check if x points are outside knots range.
    Parameters
    ----------
    x : numpy ndarray
        x values.
    y : numpy ndarray
        y values.
    knots : numpy ndarray
        Knots.
    Returns
    -------
    x : numpy ndarray
        x values all in the knots range.
    y : numpy ndarray
        y values all in the knots range.
    knots : numpy ndarray
        Unique knots.
    """
    if len(np.where(np.diff(knots) <= 0)[0]) > 0:
        knots = np.unique(knots)

    h = np.diff(knots)
    xlim1 = knots[0] - 0.01 * h[0]
    xlim2 = knots[-1] + 0.01 * h[-1]
    if x[0] < xlim1 or x[-1] > xlim2:
        P = knots[-1] - knots[0]
        x = ((x - knots[0]) % P) + knots[0]
        isort = np.argsort(x, kind="stable")
        x = x[isort]
        y = y[isort]

    return x, y, knots


def splinebase(breaks, n):
    """Generates B-spline base of order `n` for knots `breaks`.
    Parameters
    ----------
    breaks : numpy ndarray
        Knots.
    n : int
        Spline order.
    Returns
    -------
    pp : numpy ndarray
        B-spline base.
    """
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
        hl = hcopy[-1:-deg-1:-1]
        bl = breaks[0] - np.cumsum(hl)
        hr = hcopy[:deg]
        br = breaks[-1] + np.cumsum(hr)
        breaks = np.concatenate([bl[deg-1::-1], breaks, br])
        h = np.diff(breaks)
        pieces = len(h)

    coefs = np.zeros((n * pieces, n), dtype=float)
    coefs[::n, 0] = 1

    ii = np.ones((deg + 1, pieces), dtype=int)
    ii[0, :] = np.linspace(0, pieces, pieces, endpoint=False, dtype=int)
    ii = np.cumsum(ii, axis=0)
    ii[np.where(ii > pieces - 1)] = pieces - 1
    H = h[ii.flatten("F")]

    for k in range(1, n):
        for j in range(k):
            coefs[:, j] = coefs[:, j] * H / (k - j)
        Q = np.sum(coefs, axis=1)
        Q = Q.reshape((pieces, n)).T
        Q = np.cumsum(Q, axis=0)
        c0 = np.concatenate([np.zeros((1, pieces)), Q[0:deg, :]])
        coefs[:, k] = c0.flatten("F")
        fmax = np.tile(Q[n-1, :], (n, 1))
        fmax = fmax.flatten("F")
        for j in range(k + 1):
            coefs[:, j] = coefs[:, j] / fmax
        coefs[0:-deg, 0:k+1] = coefs[0:-deg, 0:k+1] - coefs[n-1:, 0:k+1]
        coefs[::n, k] = 0

    scale = np.ones(H.shape)
    for k in range(n - 1):
        scale = scale / H
        coefs[:, n-k-2] = scale * coefs[:, n-k-2]

    pieces -= 2 * deg

    ii = np.ones((deg + 1, pieces), dtype=int) * deg
    ii[0, :] = n * np.arange(1, pieces + 1, dtype=int)
    ii = np.cumsum(ii, axis=0) - 1
    coefs = coefs[ii.flatten("F"), :]

    return pp_struct(breaks0, coefs, n)


def pp_struct(br, cf, d):
    """Structure for piecewise polynomial parameters.
    Parameters
    ----------
    br : numpy ndarray
        Knots.
    cf : numpy ndarray
        Polynomial coefficients.
    d : int
        Polynomials order.
    Returns
    -------
    pp : dictionary
        Piecewise polynomial parameters structure.
    """
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


def spline_eval(pp, xx):
    """Evaluates piecewise polynomial.
    Parameters
    ----------
    pp : dictionary
        Splines base parameters.
    xx : numpy ndarray
        Evaluation points.
    Returns
    -------
    v : numpy ndarray
        Piecewise polynomial values.
    """
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
