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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('target/lapack-validation'))
    args = parser.parse_args()
    summary = args.output / 'summary.json'
    try:
        report = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(output('unexpected', f'the validation summary could not be read: {error}'))
        print(output('stale', ''))
        return 0
    unexpected = report.get('unexpected') or []
    stale = report.get('stale') or []
    unexplained = report.get('unexplained') or []
    lines = [f'{entry["family"]}: {entry["test"]}' for entry in unexpected]
    lines += [f'{family}: the family failed without a listed test, see its log' for family in unexplained]
    if report.get('error'):
        lines.insert(0, f'the run did not complete: {report["error"]}')
    print(output('unexpected', '\n'.join(lines)))
    print(output('stale', '\n'.join(f'#{entry["issue"]} {entry["test"]}' for entry in stale)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
