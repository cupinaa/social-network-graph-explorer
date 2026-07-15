# Security Policy

## Supported versions

Security fixes are applied to the latest revision of the default branch.

## Reporting a vulnerability

Report a vulnerability privately through GitHub's security advisory feature when it is available. Do not open a public issue containing exploit details or sensitive information.

The application loads Python pickle files. Pickle is not safe for untrusted input: generate these files locally with `prepare_data.py` and never load a pickle obtained from an unknown source.
