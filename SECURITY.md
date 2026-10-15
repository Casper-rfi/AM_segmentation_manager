# Security

## Reporting a problem

Report a security problem privately, not as a public issue: on GitHub, open the
**Security** tab of this repository and choose **Report a vulnerability**, or
email Casper Berger (casper.berger@rfi.ac.uk).

Say what someone would need to do to cause it, such as getting you to open a
particular file, and what it then lets them do.

## Supported versions

Fixes are made to the newest release only.

## What counts

AM-seg reads files that other people make: dataset folders, image headers, YAML
and CSV files, model files and training runs. It is a security problem if any
of these can make AM-seg:

- run code,
- write to a folder it was not asked to write to, or
- overwrite or delete a label marked corrected without asking.

AM-seg makes no network connections, except that training a new model downloads
its starting weights. Problems in the software it runs, such as torch or
AM-model-training, belong with those projects.
