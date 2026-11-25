#!/usr/bin/env python3
"""Small offline checks for selection, continuation and status accounting."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location('validation', Path(__file__).with_name('run-lapack-validation.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class Runner(unittest.TestCase):
    def test_qz_reorder_selection(self):
        self.assertEqual(v.classify("linalg::gevd::reorder::lapack::f64::lapack_qz_reorder_range"), "qz_reorder")

    def test_classification_and_empty_selection(self):
        self.assertEqual(v.classify('linalg::evd::lapack::f64::lapack_schur_cases'), 'schur')
        self.assertEqual(v.classify('linalg::evd::lapack::f64::lapack_schur_reorder_cases'), 'schur_reorder')
        self.assertEqual(v.classify('linalg::qr::lapack::c64::lapack_qr_reflector_checkers'), 'qr')
        # the transposed QR family keeps its own name and does not fall into `qr`
        self.assertEqual(v.classify('linalg::lq::lapack::f64::lapack_lq_cases'), 'lq')
        self.assertEqual(v.classify('linalg::lq::lapack::c32::lapack_lq_reference'), 'lq')
        self.assertEqual(v.classify('linalg::svd::tests::lapack_svd_f32::lapack::diagonal_families'), 'svd')
        self.assertEqual(v.classify('linalg::svd::bidiag_svd::tests::test_josef'), 'svd_large')
        self.assertIsNone(v.classify('linalg::svd::bidiag_svd::tests::test_deflation43'))
        # non-LAPACK bit-identity unit tests in the `lu` suite: ordinary tests, not a family
        self.assertIsNone(v.classify(
            'linalg::lu::lapack::c64::parallel_path_is_bit_identical_to_sequential'))
        self.assertIsNone(v.classify(
            'linalg::lu::lapack::c32::parallel_path_is_bit_identical_with_one_column_per_member'))
        # the #239 look-ahead on/off tests sit in the same module but are not case records
        self.assertIsNone(v.classify(
            'linalg::lu::lapack::c32::lookahead_does_not_change_the_factorization'))
        self.assertIsNone(v.classify(
            'linalg::lu::lapack::c64::lookahead_with_raised_threshold_does_not_change_the_factorization'))
        # a self-adjoint accuracy test beside the case suite (tensor4all/faer#100): classified, not a case record
        name = 'linalg::evd::lapack::c32::lapack_self_adjoint_evd_qr_spectrum_accuracy'
        self.assertEqual(v.classify(name), 'self_adjoint_evd')
        self.assertEqual(v.expected_records({'svd': [], 'self_adjoint_evd': [name]}), 0)
        with self.assertRaisesRegex(ValueError, 'empty'):
            v.selected_tests('')
        with self.assertRaisesRegex(ValueError, 'unclassified'):
            v.classify('linalg::evd::lapack::f64::unmapped')

    def test_continues_after_failure_timeout_and_incomplete_output(self):
        calls = []
        def fake(command, log, timeout):
            calls.append(command)
            family = log.stem
            text = 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;'
            if family == 'svd':
                text = 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out;'
            log.write_text('timeout' if family == 'lu' else text)
            return (101 if family == 'svd' else 124 if family == 'lu' else 0), 0.1
        groups = {family: [family + '_test'] for family in v.FAMILIES}
        with tempfile.TemporaryDirectory() as directory:
            rows = v.run_families('fake-binary', groups, Path(directory), fake)
        self.assertEqual(len(calls), len(v.FAMILIES))
        self.assertEqual(next(r for r in rows if r['family'] == 'svd')['exit_code'], 101)
        self.assertFalse(next(r for r in rows if r['family'] == 'lu')['complete'])
        self.assertEqual(rows[-1]['family'], 'svd_large')
        self.assertTrue(rows[-1]['complete'])
        self.assertTrue(all('--exact' in c and '--include-ignored' in c for c in calls))

    def test_svd_calibration_is_not_a_numeric_group(self):
        groups = {'svd': [
            'linalg::svd::tests::lapack_svd_f32::lapack::panics_and_errors_do_not_skip_later_cases',
            'linalg::svd::tests::lapack_svd_f32::lapack::diagonal_families'],
            'lu': ['linalg::lu::lapack::f32::lapack_lu_cases']}
        self.assertEqual(v.expected_records(groups), 2)

    def test_known_failures_keep_the_run_green_and_new_ones_do_not(self):
        known = {'linalg::svd::tests::x': 110}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'svd.log').write_text('failures:\n    linalg::svd::tests::x\n\ntest result: FAILED. 0 passed; 1 failed;\n')
            (path / 'lu.log').write_text('test result: ok. 1 passed; 0 failed;\n')
            rows = [dict(family='svd', log='svd.log'), dict(family='lu', log='lu.log')]
            groups = {'svd': ['linalg::svd::tests::x'], 'lu': ['linalg::lu::tests::y']}
            known_failing, unexpected, stale = v.classify_failures(rows, groups, known, path)
        self.assertEqual([entry['issue'] for entry in known_failing], [110])
        self.assertEqual((unexpected, stale), ([], []))

    def test_an_unlisted_failure_and_a_fixed_listed_one_are_reported(self):
        known = {'linalg::svd::tests::x': 110, 'linalg::lu::tests::y': 112}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'svd.log').write_text('failures:\n    linalg::svd::tests::new\n\ntest result: FAILED.\n')
            (path / 'lu.log').write_text('test result: ok. 1 passed; 0 failed;\n')
            rows = [dict(family='svd', log='svd.log'), dict(family='lu', log='lu.log')]
            groups = {'svd': ['linalg::svd::tests::new'], 'lu': ['linalg::lu::tests::y']}
            known_failing, unexpected, stale = v.classify_failures(rows, groups, known, path)
        self.assertEqual(unexpected, [dict(test='linalg::svd::tests::new', family='svd')])
        self.assertEqual([entry['test'] for entry in stale], ['linalg::lu::tests::y'])
        self.assertEqual(known_failing, [])

    def test_failed_tests_reads_the_summary_section_only(self):
        log = ('test linalg::a ... FAILED\n\nfailures:\n\n---- linalg::a stdout ----\n'
               'some text\n\nfailures:\n    linalg::a\n\ntest result: FAILED.\n')
        self.assertEqual(v.failed_tests(log), ['linalg::a'])

    def test_the_ledger_names_an_issue_for_every_test(self):
        # An empty ledger (no known failures on main) is the healthy state and
        # satisfies this vacuously; the load itself still checks the format.
        known = v.load_known(Path(__file__).resolve().parent.parent)
        self.assertTrue(all(isinstance(issue, int) for issue in known.values()))

    def load_ledger(self, entries):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'scripts').mkdir()
            (Path(root) / 'scripts' / v.KNOWN_FILE).write_text(json.dumps({'known': entries}))
            return v.load_known(root)

    def test_ledger_format_is_enforced_whatever_main_lists(self):
        # These cases do not depend on the repository ledger, which may be empty.
        self.assertEqual(self.load_ledger([]), {})
        self.assertEqual(self.load_ledger([{'issue': 7, 'tests': ['linalg::a', 'linalg::b']}]),
                         {'linalg::a': 7, 'linalg::b': 7})
        with self.assertRaisesRegex(ValueError, 'without an issue'):
            self.load_ledger([{'issue': None, 'tests': ['linalg::a']}])
        with self.assertRaisesRegex(ValueError, 'listed twice'):
            self.load_ledger([{'issue': 1, 'tests': ['linalg::a']}, {'issue': 2, 'tests': ['linalg::a']}])

    def test_case_counts(self):
        row, = v.case_records('LAPACK_CASES family=lu dtype=f32 planned=2 attempted=2 call_failures=1 diagnostics=1 auxiliary_attempted=4')
        self.assertEqual((row['planned'], row['attempted'], row['auxiliary_attempted']), (2, 2, 4))


class Summarizer(unittest.TestCase):
    @staticmethod
    def parse_github_output(text):
        """The `$GITHUB_OUTPUT` grammar: `name=value` lines and `name<<DELIMITER` blocks, nothing else."""
        out, lines = {}, iter(text.split('\n'))
        for line in lines:
            if not line:
                continue
            if '<<' in line.split('=', 1)[0]:
                name, delimiter = line.split('<<', 1)
                body = []
                for inner in lines:
                    if inner == delimiter:
                        break
                    body.append(inner)
                out[name] = '\n'.join(body)
            elif '=' in line:
                name, value = line.split('=', 1)
                out[name] = value
            else:
                raise ValueError(f'Invalid format {line!r}')
        return out

    def summarize(self, report, *flags):
        spec = importlib.util.spec_from_file_location('summarize', Path(__file__).with_name('summarize-lapack-failures.py'))
        s = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s)
        with tempfile.TemporaryDirectory() as root:
            if report is not None:
                (Path(root) / 'summary.json').write_text(json.dumps(report))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                old = sys.argv
                sys.argv = ['summarize', *flags, '--output', root]
                try:
                    self.assertEqual(s.main(), 0)
                finally:
                    sys.argv = old
            return buffer.getvalue() if flags else self.parse_github_output(buffer.getvalue())

    def test_several_unexpected_failures_stay_valid_step_outputs(self):
        # The 2026-11-20 nightly wrote one `family: test` line per failure and GitHub rejected
        # the second line, so the summary of exactly the failing night was lost.
        got = self.summarize({
            'unexpected': [{'family': 'lq', 'test': 'linalg::lq::lapack::c32::lapack_lq_cases'},
                           {'family': 'lq', 'test': 'linalg::lq::lapack::c64::lapack_lq_cases'}],
            'stale': [{'issue': 12, 'test': 'linalg::a'}, {'issue': 13, 'test': 'linalg::b'}],
            'unexplained': ['qr'],
        })
        self.assertEqual(got['unexpected'].split('\n'), [
            'lq: linalg::lq::lapack::c32::lapack_lq_cases',
            'lq: linalg::lq::lapack::c64::lapack_lq_cases',
            'qr: the family failed without a listed test, see its log'])
        self.assertEqual(got['stale'], '#12 linalg::a\n#13 linalg::b')

    def test_job_summary_is_plain_text_and_delimiter_never_collides(self):
        text = self.summarize({'unexpected': [{'family': 'lq', 'test': 't1'}], 'stale': []}, '--summary')
        self.assertNotIn('<<', text)
        self.assertEqual(text.split('\n')[:2], ['unexpected:', '- lq: t1'])
        spec = importlib.util.spec_from_file_location('summarize', Path(__file__).with_name('summarize-lapack-failures.py'))
        s = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s)
        first = s.output('x', '')
        delimiter = first.split('\n')[0].split('<<')[1]
        block = s.output('x', delimiter)
        self.assertNotEqual(block.split('\n')[0].split('<<')[1], delimiter)

    def test_clean_and_unreadable_summaries(self):
        self.assertEqual(self.summarize({'unexpected': [], 'stale': []}), {'unexpected': '', 'stale': ''})
        got = self.summarize(None)
        self.assertIn('could not be read', got['unexpected'])
        self.assertEqual(got['stale'], '')


if __name__ == '__main__':
    unittest.main()
