import numpy as np

#def tvfemd(x, THRESH_BWR=0.1, BSP_ORDER=26, MODES):
#	"""Time varying filter based EMD
#	Parameters
#	----------
#	x : numpy ndarray
#		Input signal x(t)
#	THRESH_BWR : float, optional
#		Instantaneous bandwidth threshold (default : 0.1)
#	BSP_ORDER : int, optional
#		b-spline order (default : 26)
#	MODES : int, optional
#		(default : )
#	Returns
#	-------
#	"""
#
#	numMAX = np.floor(np.log2(len(x))) + 1
#	MAX_IMF = numMAX + MODES
#
#	end_flag = 0
#	imf = np.zeros((MAX_IMF, len(x)), dtype=float)
#	temp_x = x.copy()
#
#	t = np.arange(1, len(x) + 1, dtype=int)
#	for nimf in range(MAX_IMF):
#		indmin_x, indmax_x = extr(temp_x)
#		if nimf == (MAX_IMF - 1):
#			imf[nimf, :] = temp_x
#			nimf += 1
#			break

		

def extr(x, t=[]):
	"""Extracts the indices of extrema
	Parameters
	----------
	x : numpy ndarray
		Input signal
	t : numpy ndarray, optional
		(default : range(len(x)))
	Returns
	-------
	indmin : numpy ndarray
		Indices of minima
	indmax : numpy ndarray
		Indices of maxima
	indzer : numpy ndarray
		Indices of zeros
	"""

	m = len(x)
	if len(t) == 0:
		t = np.arange(1, m + 1, dtype=int)

	x1 = x[:m-1]
	x2 = x[1:]
	indzer = np.where(x1 * x2 < 0.0)[0]

	if np.any(x == 0.0):
		iz = np.where(x == 0.0)[0]
		
		if np.any(np.diff(iz) == 1):
			zer = x == 0.0
			dz = np.diff(np.concatenate([[0], zer, [0]]))
			debz = np.where(dz == 1)[0]
			finz = np.where(dz == -1)[0] - 1
			indz = np.round((debz + finz) / 2)
		else:
			indz = iz

		indzer = np.sort(np.concatenate([indzer, indz]))

	d = np.diff(x)
	n = len(d)
	d1 = d[:n-1]
	d2 = d[1:]
	indmin = np.where((d1 * d2 < 0.0) & (d1 < 0.0))[0] + 1
	indmax = np.where((d1 * d2 < 0.0) & (d1 > 0.0))[0] + 1

	if any(d == 0.0):
		imax = np.array([], dtype=int)
		imin = np.array([], dtype=int)

		bad = d == 0.0
		dd = np.diff(np.concatenate([[0], bad, [0]]))
		debs = np.where(dd == 1)[0]
		fins = np.where(dd == -1)[0]

		if debs[0] == 1:
			if len(debs) > 1:
				debs = debs[1:]
				fins = fins[1:]
			else:
				debs = np.array([], dtype=int)
				fins = np.array([], dtype=int)

		if len(debs) > 0:
			if fins[-1] == m:
				if len(debs) > 1:
					debs = debs[:-1]
					fins = fins[:-1]
				else:
					debs = np.array([], dtype=int)
					fins = np.array([], dtype=int)

		lc = len(debs)
		if lc > 0:
			for k in range(lc):
				if d[debs[k]-1] > 0:
					if d[fins[k]] < 0:
						imax = np.concatenate([imax, np.round((fins[k] + debs[k]) / 2)])
				else:
					if d[fins[k]] > 0:
						imin = np.concatenate([imin, np.round((fins[k] + debs[k]) / 2)])

		if len(imax) > 0:
			indmax = np.sort(np.concatenate([indmax, imax]))
		if len(imin) > 0:
			indmin = np.sort(np.concatenate([indmin, imin]))

	print('indmin: {}'.format(indmin))
	print('indmax: {}'.format(indmax))
	print('indzer: {}'.format(indzer))