#!/usr/bin/env bash
set -euo pipefail

# ingest_rml2016.sh  --  stub for RML2016.10a ingestion
#
# NOTE: The primary RF dataset for this project is RadioML 2018.01A.
# Use scripts/ingest_rml2018.sh for acquisition.
#
# RML2016.10a is a small legacy benchmark (220K frames, 128 I/Q samples each,
# 11 classes x 20 SNRs).  It is gated behind the DeepSig portal and requires a
# manual download.  Implementation is deferred to a future branch.
#
# Manual steps if needed:
#   1. Create an account at https://deepsig.io and request dataset access.
#   2. Download RML2016.10a.tar.bz2 (or the .pkl directly).
#   3. Place the file at:
#        <DATA_DIR>/rml2016/RML2016.10a_dict.pkl
#      where DATA_DIR defaults to data/ in the repo root.
#
# Expected directory structure after manual placement:
#   data/
#     rml2016/
#       RML2016.10a_dict.pkl   -- 11 classes x 20 SNRs x 1000 frames, shape (2, 128) each

echo "[ingest_rml2016] RML2016.10a ingestion is not yet implemented." >&2
echo "[ingest_rml2016] For RF data acquisition use: bash scripts/ingest_rml2018.sh" >&2
echo "[ingest_rml2016] See the comment block in this script for manual download instructions." >&2
exit 1
