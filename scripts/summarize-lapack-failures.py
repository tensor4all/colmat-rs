#!/usr/bin/env python3
"""Print the run's new and stale failures as GitHub Actions step outputs.

Reads target/lapack-validation/summary.json (written by run-lapack-validation.py) and emits
`unexpected` and `stale` outputs, which the failure reporter puts in its issue. Missing or
unreadable evidence is reported as a line the reporter shows, never as silence.

The outputs are written in the multiline form (`name<<DELIMITER`, the value, `DELIMITER`):
a value is one `family: test` line per failure, and a bare `name=value` line would end at the
first newline and leave the next line as an invalid `$GITHUB_OUTPUT` entry.
"""
import argparse
import json
import uuid
from pathlib import Path


def output(name, value):
    """One step output in the multiline form, with a delimiter the value cannot contain."""
    delimiter = f'ghadelimiter_{uuid.uuid4().hex}'
    while delimiter in value:
        delimiter = f'ghadelimiter_{uuid.uuid4().hex}'
    return f'{name}<<{delimiter}\n{value}\n{delimiter}'


def render(name, value, summary):
    if not summary:
        return output(name, value)
    return f'{name}:' + ''.join(f'\n- {line}' for line in value.split('\n') if line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('target/lapack-validation'))
    parser.add_argument('--summary', action='store_true',
                        help='print readable text for a job summary instead of step outputs')
    args = parser.parse_args()
    summary = args.output / 'summary.json'
    try:
        report = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(render('unexpected', f'the validation summary could not be read: {error}', args.summary))
        print(render('stale', '', args.summary))
        return 0
    unexpected = report.get('unexpected') or []
    stale = report.get('stale') or []
    unexplained = report.get('unexplained') or []
    lines = [f'{entry["family"]}: {entry["test"]}' for entry in unexpected]
    lines += [f'{family}: the family failed without a listed test, see its log' for family in unexplained]
    if report.get('error'):
        lines.insert(0, f'the run did not complete: {report["error"]}')
    print(render('unexpected', '\n'.join(lines), args.summary))
    print(render('stale', '\n'.join(f'#{entry["issue"]} {entry["test"]}' for entry in stale), args.summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
