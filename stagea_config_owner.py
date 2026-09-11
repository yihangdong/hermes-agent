"""Cold-safe owner of Stage-A's already-bounded literal config bytes.

The caller owns confinement, managed-presence refusal, byte limits and value
validation. This dedicated parser performs no file, environment, discovery,
default, normalization or cache work; ordinary config loaders remain separate.
"""

import yaml


def parse_stagea_task_config_bytes(raw: bytes):
    """Parse once without expanding or merging; propagate malformed YAML."""
    if type(raw) is not bytes:
        raise TypeError("Stage-A config parser requires bounded bytes")
    return yaml.safe_load(raw)
