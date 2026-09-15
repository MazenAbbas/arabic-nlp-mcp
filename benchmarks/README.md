# Benchmarks

`performance.py` measures local preprocessing throughput after a warm-up. It is
not an NLP accuracy benchmark and its output must include the machine and Python
version when published.

The test suite contains labeled regression and metamorphic cases. Those cases
protect known behavior but must not be presented as independent accuracy data.
Any future accuracy benchmark must pin the dataset version, license, split,
metric implementation, dependency versions, and random seed.
