#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck disable=SC1090
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"
mkdir -p "$DOWNLOAD_DIR"/{wikimedia,tatoeba,opus}
cd "$PACK_DIR"
fetch() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  echo "[fetch] $url -> $out"
  curl -L --fail --retry 5 --retry-delay 5 --continue-at - -o "$out" "$url"
  sha256sum "$out" | tee -a reports/downloads.sha256
}
mkdir -p reports
fetch https://dumps.wikimedia.org/bewiki/latest/bewiki-latest-pages-articles.xml.bz2 "$DOWNLOAD_DIR/wikimedia/bewiki-latest-pages-articles.xml.bz2"
fetch https://dumps.wikimedia.org/be_x_oldwiki/latest/be_x_oldwiki-latest-pages-articles.xml.bz2 "$DOWNLOAD_DIR/wikimedia/be_x_oldwiki-latest-pages-articles.xml.bz2"
fetch https://dumps.wikimedia.org/bewikisource/latest/bewikisource-latest-pages-articles.xml.bz2 "$DOWNLOAD_DIR/wikimedia/bewikisource-latest-pages-articles.xml.bz2"
fetch https://dumps.wikimedia.org/bewiktionary/latest/bewiktionary-latest-pages-articles.xml.bz2 "$DOWNLOAD_DIR/wikimedia/bewiktionary-latest-pages-articles.xml.bz2"
fetch https://dumps.wikimedia.org/bewikibooks/latest/bewikibooks-latest-pages-articles.xml.bz2 "$DOWNLOAD_DIR/wikimedia/bewikibooks-latest-pages-articles.xml.bz2"
fetch https://downloads.tatoeba.org/exports/sentences.tar.bz2 "$DOWNLOAD_DIR/tatoeba/sentences.tar.bz2"
fetch https://downloads.tatoeba.org/exports/links.tar.bz2 "$DOWNLOAD_DIR/tatoeba/links.tar.bz2"
# OPUS URLs can move by version; keep failures non-fatal and verify reports.
fetch 'https://opus.nlpl.eu/download.php?f=Tatoeba/v2023-04-12/moses/be-en.txt.zip' "$DOWNLOAD_DIR/opus/Tatoeba.be-en.txt.zip" || echo '[warn] OPUS Tatoeba URL needs verification'
fetch 'https://opus.nlpl.eu/download.php?f=JW300/v1/moses/be-en.txt.zip' "$DOWNLOAD_DIR/opus/JW300.be-en.txt.zip" || echo '[warn] OPUS JW300 URL needs verification'
echo "[done] downloads complete. Next: run repo extractor/filter, LID, dedup and license manifest before training."
