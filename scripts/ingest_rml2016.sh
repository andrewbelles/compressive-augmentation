#!/usr/bin/env bash
set -euo pipefail

# ingest_rml2016.sh  --  stub for RML2016.10a ingestion
#
# RML2016.10a is gated behind the DeepSig portal and requires a manual download.
# This script will be implemented in an rf/* branch.
#
# Manual steps until this script is complete:
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

echo "[ingest_rml2016] RML2016.10a requires a manual download from the DeepSig portal." >&2
echo "[ingest_rml2016] See the comment block in this script for instructions." >&2
echo "[ingest_rml2016] Automated ingest will be implemented in an rf/* branch." >&2
exit 1
