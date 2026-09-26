# Source provenance

FissionDB starts from the current Mangrove product engine snapshot `ce053253382331f2ce30062be7357e5f67cda84c`, exported on 2026-09-26.

The initial commit contains the complete product sources, tests, build and packaging files, specifications, documentation and validation reports. Prior Git history, datasets, credentials and generated binaries are not imported.

Native and Python sources and all tests are byte-identical to that snapshot. The repository title and links are updated for FissionDB. CI invokes the existing `make test` target to supply the Python path used by subprocess tests. Existing package, command and on-disk format names remain compatible.

The original Apache-2.0 license and attribution are preserved. Historical measurement reports retain their original names, paths and methodology; the repository creation itself is not a new benchmark run.
