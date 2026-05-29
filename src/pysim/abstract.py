import logging

import numpy as np
import scipy
import scipy.linalg

logger = logging.getLogger(__name__)


class AbstractPySim:
    def __init__(
        self,
        *,
        wavelength=22,
        halfdriver_factor=0.962,
        nsegs=101,
        rcond=1e-16,
        nsmallest=0,
        run_iterative_improvement=False,
        run_svd=False,
    ):
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.nsegs = nsegs
        self.rcond = rcond
        self.nsmallest = nsmallest

        self.eps = 8.8541878188e-12
        self.mu = 1.25663706127e-6
        self.c = 1 / np.sqrt(self.eps * self.mu)

        self.freq = self.c / self.wavelength  # meters/sec / meters = 1/sec = Hz
        self.omega = self.freq * 2 * np.pi  # radians/sec

        """
        self.k = np.pi*2/self.wavelength      #radians/meter
        self.k = np.pi*2/(self.c/self.freq)
        """
        self.k = self.omega / self.c

        self.jomega = (0 + 1j) * self.omega  # imaginary radians/sec
        self.wire_radius = 0.0005
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        self.driver_seg_idx = self.nsegs // 2

        self.run_svd = run_svd
        self.run_iterative_improvement = run_iterative_improvement

    @staticmethod
    def solve_using_svd(A, b, rcond=1e-16, nsmallest=0):
        u, s, vh = scipy.linalg.svd(A)
        abss = np.abs(s)

        if nsmallest > 0:
            # sorted in decreasing order?
            assert np.all(abss[1:] <= abss[:-1])
            logger.debug("abss=%s, abss[-nsmallest]=%s", abss, abss[-nsmallest])
            mask = abss <= abss[-nsmallest]
        else:
            mask = abss > rcond * np.max(abss)

        logger.debug(
            "condition=%.3e, kept=%d",
            np.max(abss) / np.min(abss),
            np.count_nonzero(mask),
        )

        u, s, vh = u[:, mask], s[mask], vh[mask, :]

        def solve(b):
            return vh.conj().T @ (np.diag(1 / s) @ (u.T @ b))

        x = solve(b)

        if False:
            x = np.array(x, dtype=np.complex256)

            r = b - A @ x
            logger.debug("svd residual norm (0): %.3e", np.linalg.norm(r))
            x += solve(r)
            r = b - A @ x
            logger.debug("svd residual norm (1): %.3e", np.linalg.norm(r))
            x += solve(r)
            r = b - A @ x
            logger.debug("svd residual norm (2): %.3e", np.linalg.norm(r))

        return x

    def factor_and_solve(self):
        factors = scipy.linalg.lu_factor(self.z)

        v = np.zeros(shape=(self.z.shape[0],), dtype=np.complex128)
        v[self.driver_seg_idx] = 1

        if self.run_svd:
            i_svd = self.solve_using_svd(
                self.z, v, rcond=self.rcond, nsmallest=self.nsmallest
            )

            r = v - np.dot(self.z, i_svd)
            logger.debug("i_svd error (0): %.3e", np.linalg.norm(r))

        i = scipy.linalg.lu_solve(factors, v)

        if self.run_iterative_improvement:
            i = np.array(i, dtype=np.complex256)
            r = v - np.dot(self.z, i)
            logger.debug("i error (0): %.3e", np.linalg.norm(r))
            i += scipy.linalg.lu_solve(factors, r)

            r = v - np.dot(self.z, i)
            logger.debug("i error (1): %.3e", np.linalg.norm(r))
            i += scipy.linalg.lu_solve(factors, r)

            r = v - np.dot(self.z, i)
            logger.debug("i error (2): %.3e", np.linalg.norm(r))

        if self.run_svd:
            logger.debug("error vs. svd: %.3e", np.linalg.norm(i_svd - i))

        driver_impedance = v[self.driver_seg_idx] / i[self.driver_seg_idx]
        logger.debug(
            "driver |Z|=%.4f phase=%.2f deg",
            np.abs(driver_impedance),
            np.angle(driver_impedance) * 180 / np.pi,
        )

        if self.run_svd:
            driver_impedance_svd = v[self.driver_seg_idx] / i_svd[self.driver_seg_idx]
            logger.debug(
                "driver (svd) |Z|=%.4f phase=%.2f deg",
                np.abs(driver_impedance_svd),
                np.angle(driver_impedance_svd) * 180 / np.pi,
            )

        if self.run_svd:
            return driver_impedance, (i, i_svd)
        else:
            return driver_impedance, i
