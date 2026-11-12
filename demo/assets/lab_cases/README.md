# Diagnostic reference images

These 18 original JPEGs come from the group-provided RDD2022 archives: Czech.zip, India.zip and United_States.zip.
Six images per country are selected deterministically from the existing test/domain_test manifests, with available class coverage.
They are a small teaching/diagnostic subset, not a representative evaluation set. Training and dataset partitions are unchanged.

manifest.json preserves country, source split, original image/XML archive member names, canonical target boxes, image SHA256 and annotation SHA256.
JPEG bytes are copied without editing. Live overlays and challenges are generated at runtime and never overwrite these originals.
See ../../../data/README.md for dataset provenance. Rebuild with scripts/prepare_lab_cases.py and the original group archives.
