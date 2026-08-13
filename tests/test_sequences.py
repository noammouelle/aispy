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

class TestLMTResidualSeparation:
    """The LMT blocks leave a residual vertical arm separation of order
    (hbar k/m) * dt_pi * n(n-1)/2. At n=101 and a 10 kHz Rabi frequency that is
    3.3 mm -- larger than a typical coherence length, so nothing interferes and
    the run silently returns no fringe."""

    def test_warns_when_separation_exceeds_coherence_length(self):
        # n = 101 at 10 kHz: the case from the coriolis commit message
        params = base_params(
            loopnumber=1, lmt_order=101,
            rabi_freq=2 * 3.141592653589793 * 1e4,
            coherencelength=1e-4,          # 0.1 mm, below the ~3.3 mm residual
            interrogation_time=[0.05],
        )
        with pytest.warns(RuntimeWarning, match='will not interfere'):
            write_sequence(params)

    def test_silent_when_pulse_is_short_enough(self):
        # same sequence at 100 kHz brings the residual to ~0.34 mm
        params = base_params(
            loopnumber=1, lmt_order=101,
            rabi_freq=2 * 3.141592653589793 * 1e5,
            coherencelength=1e-2,          # 10 mm
            interrogation_time=[0.05],
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            write_sequence(params)         # must not raise

    def test_scales_with_pulse_duration(self):
        """Residual separation scales with dt_pi, i.e. inversely with rabi_freq.
        Halving the pulse duration must move a warning case to a silent one."""
        slow = base_params(loopnumber=1, lmt_order=51,
                           rabi_freq=2 * 3.141592653589793 * 1e4,
                           coherencelength=3e-4,
                           interrogation_time=[0.05])
        with pytest.warns(RuntimeWarning):
            write_sequence(slow)

        fast = base_params(loopnumber=1, lmt_order=51,
                           rabi_freq=2 * 3.141592653589793 * 1e6,
                           coherencelength=3e-4,
                           interrogation_time=[0.05])
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            write_sequence(fast)


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
