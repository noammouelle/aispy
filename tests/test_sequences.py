"""Tests for the AISFlow sequence builder.

The builder is where every silent non-closure in this codebase has originated:
it decides the pulse schedule, the LMT boundary corrections and the path list,
and its failure mode is not an exception but an interferometer that quietly does
not interfere. These tests cover the cases that have actually bitten.

Run with:  pytest tests/ -v
"""

import os
import sys
import tempfile
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aispy.utils import AISFlow, kz, m, hbar, pi  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def base_params(loopnumber=1, interrogation_time=None, lmt_order=1,
                rabi_freq=2 * 3.141592653589793 * 1e4, coherencelength=1.0,
                wtype='gaussian', **pulse_overrides):
    """A minimal but complete parameter dict for an ultranarrow MZ sequence."""
    if interrogation_time is None:
        # the builder wants ceil(loopnumber/2) entries
        interrogation_time = [0.005] * ((loopnumber + 1) // 2)

    pulse_params = {
        'rabi_freq': rabi_freq,
        'wtype': wtype,
        'phi0': 0.0,
        'kx_psr': 0.0,
        'ky_psr': 0.0,
        'beam_radius': 0.05,
        'baseline': 0.0,
        'waist': 0.03,
        'focallength': 0.0,
    }
    pulse_params.update(pulse_overrides)

    return {
        'cloud_params': {
            'natoms': 1, 'initialstate': 0, 'sigma': 0.0,
            'transtemp': 0.0, 'longtemp': 0.0,
            'x0': [0.0, 0.0, 0.0], 'v0': [0.0, 0.0, 0.0],
        },
        'potential_params': {'utype': 'linear_pot'},
        'sequence_params': {
            'sequencename': 'MZ',
            'automaticdetuning': 1,
            'ultranarrow': 1,
            'loopnumber': loopnumber,
            'interrogation_time': interrogation_time,
            'lmt_order': lmt_order,
            'dt_lmt': 1e-6,
            't_init': 0.0,
            'detectiontime': 0.05,
            'frequencychirp': 0,
            'kchirp': 0,
        },
        'simulation_params': {
            'amplitudethreshold': 0.001,
            'coherencelength': coherencelength,
            'usemcbranching': 0, 'usepathselection': 1,
            'usestaticapprox': 0, 'ultrafast': 1,
            'ignoredetuning': 0, 'seed': -1,
            'usedetvolselection': 0,
            'xdet': [-1.0, 1.0], 'ydet': [-1.0, 1.0], 'zdet': [-1.0, 1.0],
            'gslqagabserr': 1e-12, 'gslqagrelerr': 1e-12,
            'gslkinodeabserr': 1e-9, 'gslkinoderelerr': 0,
            'gslpulseodeabserr': 1e-9, 'gslpulseoderelerr': 0,
        },
        'io_params': {'printprobs': 1, 'printwavepackets': 0},
        'pulse_params': pulse_params,
    }


def write_sequence(params):
    """Run AISFlow into a temp dir and return the generated .aisi text."""
    with tempfile.TemporaryDirectory() as workdir:
        AISFlow(params, flowdir='TEST', workdir=workdir)
        with open(os.path.join(workdir, 'TEST.aisi')) as fh:
            return fh.read()


def parse_arrays(text):
    """Parse an .aisi file into {key: [tokens]}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        out[parts[0]] = parts[1:]
    return out


# --------------------------------------------------------------------------
# loopnumber vs len(interrogation_time)
# --------------------------------------------------------------------------

class TestLoopCountConsistency:
    """`loopnumber` and `len(interrogation_time)` are different quantities that
    both feed the geometry. Before v0.0.2 a mismatch produced a pulse schedule
    and a path list describing different interferometers, with no error."""

    @pytest.mark.parametrize('loopnumber,n_times', [
        (1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3),
    ])
    def test_accepts_correct_pairing(self, loopnumber, n_times):
        params = base_params(loopnumber=loopnumber,
                             interrogation_time=[0.005] * n_times)
        text = write_sequence(params)
        assert 'pathstosimulate' in text

    @pytest.mark.parametrize('loopnumber,n_times', [
        (2, 2),   # the natural mistake: one time per loop
        (4, 4),
        (3, 1),
        (4, 1),
        (1, 2),
    ])
    def test_rejects_mismatch(self, loopnumber, n_times):
        params = base_params(loopnumber=loopnumber,
                             interrogation_time=[0.005] * n_times)
        with pytest.raises(AssertionError, match='interrogation_time'):
            write_sequence(params)

    @pytest.mark.parametrize('loopnumber', [1, 2, 3, 4])
    def test_diamond_count_matches_loopnumber(self, loopnumber):
        """The palindrome must produce exactly `loopnumber` diamonds."""
        params = base_params(loopnumber=loopnumber)
        # the internal assertion D == ndiamonds would fire if not
        write_sequence(params)


# --------------------------------------------------------------------------
# LMT residual separation vs coherence length
# --------------------------------------------------------------------------

class TestNoFalseLMTSeparationWarning:
    """The LMT ladder drift is compensated, and must not be warned about.

    A warning used to fire here claiming that the blocks leave a residual arm
    separation of (hbar k/m)*dt_pi*n(n-1)/2 and that the arms would not
    interfere.  That formula measures the drift as if nothing corrected it,
    while ``_write_ultranarrow_MZ_Lloops`` has already corrected it with the
    delta = (n-1)*u/2 carve-outs on the outer dead times.

    Checked against ais++, which does enforce ``coherencelength``: at n = 1001
    with 500 us pulses it interferes at 1e-8 m and stops at 1e-9 m, so the real
    separation is ~1.5 nm rather than the 1.6 m the formula predicted.  The
    cases below are the ones the old guard warned about; every one of them
    interferes on every record in ais++.
    """

    @pytest.mark.parametrize('lmt_order, rabi_hz, coherencelength', [
        (101, 1e4, 1e-4),      # the old guard's own motivating case
        (51, 1e4, 3e-4),
        (1001, 1e3, 1.0),      # 500 us pulses, where it claimed 1.6 m
    ])
    def test_no_warning_for_sequences_that_do_interfere(
            self, lmt_order, rabi_hz, coherencelength):
        params = base_params(
            loopnumber=1, lmt_order=lmt_order,
            rabi_freq=2 * 3.141592653589793 * rabi_hz,
            coherencelength=coherencelength,
            interrogation_time=[0.05 if lmt_order < 1001 else 2.225],
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            write_sequence(params)


# --------------------------------------------------------------------------
# LMT position closure
# --------------------------------------------------------------------------

class TestLMTPositionClosure:
    """The init and final LMT acceleration blocks each accumulate
    Δz = (hbar k/m)·dt·n(n-1)/2 of arm separation in the same direction. The
    builder cancels this by shortening the dead time at each end by
    δ_half = (n-1)(dt_pi+dt_lmt)/2.

    That correction previously carried a (-1)**D factor on the last dead time,
    which LENGTHENED it for odd D instead of shortening it, leaving a residual
    of 2·δ_half·n·v_rec — 6.7 mm at n=101 and 41 mm at n=251, against a ~1 mm
    interference tolerance. Every odd-loop LMT sequence failed to close; even
    loop counts were fine, which is exactly the kind of asymmetry that hides.

    These tests read the correction back out of the generated schedule, so the
    sign cannot silently flip again.
    """

    @staticmethod
    def _dead_times(arrays):
        """Gaps between consecutive pulses, from the generated schedule."""
        t0 = [float(v) for v in arrays['t0']]
        t1 = [float(v) for v in arrays['t1']]
        return [t0[i + 1] - t1[i] for i in range(len(t0) - 1)]

    @staticmethod
    def _interrogation_gaps(arrays):
        """The interrogation dead times only; the LMT blocks contribute
        dt_lmt-sized gaps that would swamp the comparison.

        Structure of the result, which the tests below rely on: no pulse sits
        between the second dead time of one diamond and the first of the next,
        so those two MERGE into a single gap of ~2*T_eff. The list is therefore

            [T_eff - delta_half,  2*T_eff, ..., 2*T_eff,  T_eff - delta_half - extra_last]

        i.e. the first and last entries are single, corrected dead times and the
        middles are uncorrected pairs. The cutoff must sit below the single-gap
        size, not half the merged size, or the boundaries are silently dropped.
        """
        gaps = TestLMTPositionClosure._dead_times(arrays)
        cutoff = 0.25 * max(gaps)
        return [g for g in gaps if g > cutoff]

    @staticmethod
    def _delta_half(params, n):
        rabi = params['pulse_params']['rabi_freq']
        dt_pi = float(pi / rabi)
        return (n - 1) * (dt_pi + params['sequence_params']['dt_lmt']) / 2

    @pytest.mark.parametrize('loopnumber', [1, 3, 5])
    def test_odd_D_boundary_gaps_are_equal(self, loopnumber):
        """For odd D the first and last interrogation dead times must be equal:
        both are shortened by delta_half and extra_last is zero.

        This is the direct signature of the bug. With the (-1)**D factor the
        last gap was LENGTHENED by delta_half while the first was shortened, so
        the two differed by exactly 2*delta_half.
        """
        n = 11
        params = base_params(loopnumber=loopnumber, lmt_order=n,
                             interrogation_time=[0.05] * ((loopnumber + 1) // 2))
        gaps = self._interrogation_gaps(parse_arrays(write_sequence(params)))
        delta_half = self._delta_half(params, n)

        assert gaps[0] == pytest.approx(gaps[-1], abs=1e-9), (
            f"D={loopnumber}: first gap {gaps[0]:.9g} != last gap {gaps[-1]:.9g}; "
            f"they differ by {abs(gaps[0]-gaps[-1]):.6g} s and 2*delta_half is "
            f"{2*delta_half:.6g} s -- this is the (-1)**D sign bug"
        )

    @pytest.mark.parametrize('loopnumber', [3, 4, 5, 6])
    def test_boundary_gaps_are_shorter_than_middle_gaps(self, loopnumber):
        """Only the first and last interrogation dead times carry the
        correction, so they must be delta_half shorter than the middle ones.
        Compared within a single schedule, so the separate reduction of
        interrogation_time by the LMT block duration cancels out."""
        n = 11
        params = base_params(loopnumber=loopnumber, lmt_order=n,
                             interrogation_time=[0.05] * ((loopnumber + 1) // 2))
        gaps = self._interrogation_gaps(parse_arrays(write_sequence(params)))
        delta_half = self._delta_half(params, n)

        middles = gaps[1:-1]
        assert middles, "expected uncorrected middle dead times for D >= 3"
        # middles are merged pairs, so halve them to get one uncorrected dead time
        uncorrected = max(middles) / 2.0

        assert uncorrected - gaps[0] == pytest.approx(delta_half, rel=0.02), (
            f"D={loopnumber}: first gap is {uncorrected - gaps[0]:.6g} s shorter "
            f"than an uncorrected dead time, expected delta_half = {delta_half:.6g} s"
        )
        # the last gap carries delta_half plus, for even D, extra_last
        assert uncorrected - gaps[-1] >= delta_half * 0.98, (
            f"D={loopnumber}: last gap was not shortened by at least delta_half"
        )

    @pytest.mark.parametrize('n', [11, 101, 251])
    def test_correction_scales_with_lmt_order(self, n):
        """delta_half grows as (n-1), so the shortening must too. The old bug's
        residual grew as n^2, which is why it stayed invisible at low n."""
        params = base_params(loopnumber=3, lmt_order=n,
                             interrogation_time=[0.3, 0.3])
        gaps = self._interrogation_gaps(parse_arrays(write_sequence(params)))
        delta_half = self._delta_half(params, n)

        shortening = max(gaps[1:-1]) / 2.0 - gaps[0]
        assert shortening == pytest.approx(delta_half, rel=0.02), (
            f"n={n}: shortening {shortening:.6g} != delta_half {delta_half:.6g}"
        )

    @pytest.mark.parametrize('loopnumber', [2, 4, 6])
    def test_even_D_gets_the_extra_correction(self, loopnumber):
        """Even D needs a further 2*delta_half/n on the last dead time, because
        the init and final block displacements add rather than cancel. Odd D
        must NOT get it."""
        n = 11
        params = base_params(loopnumber=loopnumber, lmt_order=n,
                             interrogation_time=[0.05] * ((loopnumber + 1) // 2))
        gaps = self._interrogation_gaps(parse_arrays(write_sequence(params)))
        delta_half = self._delta_half(params, n)
        extra_last = 2 * delta_half / n

        assert gaps[0] - gaps[-1] == pytest.approx(extra_last, rel=0.02), (
            f"D={loopnumber}: last gap is {gaps[0]-gaps[-1]:.6g} s shorter than "
            f"the first, expected extra_last = {extra_last:.6g} s"
        )

    def test_schedule_stays_monotonic_at_high_lmt(self):
        """The correction subtracts from dead times, so at large n it must not
        drive a dead time negative and reorder the schedule."""
        params = base_params(loopnumber=4, lmt_order=101,
                             interrogation_time=[0.2, 0.2])
        arrays = parse_arrays(write_sequence(params))
        t0 = [float(v) for v in arrays['t0']]
        t1 = [float(v) for v in arrays['t1']]
        for i in range(len(t0) - 1):
            assert t0[i + 1] >= t1[i] - 1e-18, (
                f"pulse {i+1} starts before pulse {i} ends: the closure "
                f"correction over-subtracted"
            )


# --------------------------------------------------------------------------
# schedule well-formedness
# --------------------------------------------------------------------------

class TestScheduleWellFormed:

    @pytest.mark.parametrize('loopnumber', [1, 2, 3, 4])
    def test_pulse_arrays_have_equal_length(self, loopnumber):
        """Every per-pulse array must have the same length; ais++ indexes them
        together and a short one reads past the end."""
        arrays = parse_arrays(write_sequence(base_params(loopnumber=loopnumber)))
        per_pulse = ['t0', 't1', 'kx', 'ky', 'kz', 'omega', 'rabifreq', 'wtype',
                     'phi0', 'waist', 'focallength', 'zlaser', 'beamradius',
                     'baseline', 'kxchirp', 'kychirp', 'kzchirp', 'frequencychirp']
        lengths = {key: len(arrays[key]) for key in per_pulse if key in arrays}
        assert len(set(lengths.values())) == 1, f"ragged pulse arrays: {lengths}"

    @pytest.mark.parametrize('loopnumber', [1, 2, 3, 4])
    def test_pulses_are_ordered_and_non_overlapping(self, loopnumber):
        arrays = parse_arrays(write_sequence(base_params(loopnumber=loopnumber)))
        t0 = [float(v) for v in arrays['t0']]
        t1 = [float(v) for v in arrays['t1']]
        for i, (a, b) in enumerate(zip(t0, t1)):
            assert b > a, f"pulse {i} ends before it starts"
        for i in range(len(t0) - 1):
            assert t0[i + 1] >= t1[i] - 1e-18, f"pulse {i+1} overlaps pulse {i}"

    @pytest.mark.parametrize('loopnumber', [1, 2, 3, 4])
    def test_detection_is_after_last_pulse(self, loopnumber):
        params = base_params(loopnumber=loopnumber)
        arrays = parse_arrays(write_sequence(params))
        assert float(params['sequence_params']['detectiontime']) >= float(arrays['t1'][-1])

    @pytest.mark.parametrize('loopnumber', [1, 2, 3, 4])
    def test_four_paths_are_written(self, loopnumber):
        arrays = parse_arrays(write_sequence(base_params(loopnumber=loopnumber)))
        assert len(arrays['pathstosimulate']) == 4

    @pytest.mark.parametrize('loopnumber', [1, 2, 3, 4])
    def test_path_length_matches_pulse_count(self, loopnumber):
        """Each character of a path string is one beam-splitter interaction, so
        the strings must be exactly one longer than the pulse count."""
        arrays = parse_arrays(write_sequence(base_params(loopnumber=loopnumber)))
        n_pulses = len(arrays['t0'])
        for path in arrays['pathstosimulate']:
            assert len(path) == n_pulses + 1, (
                f"path {path!r} has length {len(path)}, expected {n_pulses + 1}"
            )

    def test_odd_lmt_order_is_required(self):
        with pytest.raises(AssertionError, match='odd lmt_order'):
            write_sequence(base_params(lmt_order=2))

    def test_interrogation_time_must_fit_lmt_blocks(self):
        with pytest.raises(AssertionError, match='too short'):
            write_sequence(base_params(lmt_order=51, interrogation_time=[1e-6]))


# --------------------------------------------------------------------------
# wavefront parameters
# --------------------------------------------------------------------------

class TestWavefrontParams:

    def test_zernike_coefficients_are_written(self):
        params = base_params(zernike_coeffs={4: 0.1, 11: -0.02})
        arrays = parse_arrays(write_sequence(params))
        n_pulses = len(arrays['t0'])
        assert arrays['zernikecoeff_4'] == ['0.1'] * n_pulses
        assert arrays['zernikecoeff_11'] == ['-0.02'] * n_pulses

    def test_no_zernike_keys_when_unset(self):
        arrays = parse_arrays(write_sequence(base_params()))
        assert not [k for k in arrays if k.startswith('zernikecoeff')]

    def test_interpolated_beam_writes_file_path(self):
        params = base_params(wtype='interpolated', beam_file='/tmp/beam.h5')
        arrays = parse_arrays(write_sequence(params))
        n_pulses = len(arrays['t0'])
        assert arrays['beaminterpolationparamsfilenames'] == ['/tmp/beam.h5'] * n_pulses
        assert arrays['wtype'] == ['interpolated'] * n_pulses

    def test_interpolated_beam_requires_a_file(self):
        with pytest.raises(ValueError, match='requires'):
            write_sequence(base_params(wtype='interpolated'))

    def test_beam_file_without_interpolated_wtype_is_rejected(self):
        """Silently ignoring the file would produce an analytic-beam result the
        caller believes came from their grid."""
        with pytest.raises(ValueError, match='would be ignored'):
            write_sequence(base_params(wtype='gaussian', beam_file='/tmp/beam.h5'))

    def test_zernike_with_interpolated_is_rejected(self):
        with pytest.raises(ValueError, match='cannot be combined'):
            write_sequence(base_params(wtype='interpolated', beam_file='/tmp/b.h5',
                                       zernike_coeffs={4: 0.1}))

    def test_tiptilt_is_written_for_interpolated_beams(self):
        params = base_params(wtype='interpolated', beam_file='/tmp/beam.h5',
                             tiptiltx=0.5, tiptilty=-0.25)
        arrays = parse_arrays(write_sequence(params))
        n_pulses = len(arrays['t0'])
        assert arrays['tiptiltx'] == ['0.5'] * n_pulses
        assert arrays['tiptilty'] == ['-0.25'] * n_pulses

    def test_tiptilt_on_analytic_beam_is_rejected(self):
        with pytest.raises(ValueError, match='only applied'):
            write_sequence(base_params(wtype='gaussian', tiptiltx=0.5))

    def test_no_tiptilt_keys_when_zero(self):
        arrays = parse_arrays(write_sequence(base_params()))
        assert 'tiptiltx' not in arrays


# --------------------------------------------------------------------------
# rotating frame plumbing
# --------------------------------------------------------------------------

class TestRotatingFrame:

    def test_rotation_key_is_written_when_given(self):
        params = base_params()
        params['potential_params'] = {
            'utype': 'rotating_linear_pot',
            'rotation': [0.0, 5.156e-5, 5.156e-5],
        }
        arrays = parse_arrays(write_sequence(params))
        assert arrays['utype'] == ['rotating_linear_pot']
        assert [float(v) for v in arrays['rotation']] == [0.0, 5.156e-5, 5.156e-5]

    def test_rotation_key_absent_for_inertial_runs(self):
        """ais++ rejects `rotation` alongside an inertial utype, so it must not
        be emitted by default."""
        arrays = parse_arrays(write_sequence(base_params()))
        assert 'rotation' not in arrays


# --------------------------------------------------------------------------
# unknown sequences
# --------------------------------------------------------------------------

def test_unknown_sequence_name_raises_value_error():
    """'RB' used to dispatch to methods that were never defined, so it raised
    AttributeError instead of anything informative."""
    params = base_params()
    params['sequence_params']['sequencename'] = 'RB'
    with pytest.raises(ValueError, match='Unknown sequencename'):
        write_sequence(params)
