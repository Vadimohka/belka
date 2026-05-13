#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/downloads"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/downloads/download_owner_${TS}.log"
STATUS_FILE="$REPORT_DIR/downloads/download_status.jsonl"
SHA256_FILE="$REPORT_DIR/downloads/download_v2_sha256_${TS}.txt"

exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA OWNER DOWNLOAD — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "PACK_DIR=$PACK_DIR"
echo ""

bash local/repo_guard.sh
echo ""

# ---------- source list ----------
sources=(
  "bewiki_articles|Wikimedia|bewiki|https://dumps.wikimedia.org/bewiki/latest/bewiki-latest-pages-articles-multistream.xml.bz2|wikimedia/bewiki-latest-pages-articles-multistream.xml.bz2|CC_BY-SA_GFDL"
  "bewiki_index|Wikimedia|bewiki_index|https://dumps.wikimedia.org/bewiki/latest/bewiki-latest-pages-articles-multistream-index.txt.bz2|wikimedia/bewiki-latest-pages-articles-multistream-index.txt.bz2|CC_BY-SA_GFDL"
  "be_x_oldwiki_articles|Wikimedia|be_x_oldwiki|https://dumps.wikimedia.org/be_x_oldwiki/latest/be_x_oldwiki-latest-pages-articles-multistream.xml.bz2|wikimedia/be_x_oldwiki-latest-pages-articles-multistream.xml.bz2|CC_BY-SA_GFDL"
  "bewikisource_articles|Wikimedia|bewikisource|https://dumps.wikimedia.org/bewikisource/latest/bewikisource-latest-pages-articles-multistream.xml.bz2|wikimedia/bewikisource-latest-pages-articles-multistream.xml.bz2|CC_BY-SA_GFDL"
  "bewiktionary_articles|Wikimedia|bewiktionary|https://dumps.wikimedia.org/bewiktionary/latest/bewiktionary-latest-pages-articles-multistream.xml.bz2|wikimedia/bewiktionary-latest-pages-articles-multistream.xml.bz2|CC_BY-SA_GFDL"
  "bewikibooks_articles|Wikimedia|bewikibooks|https://dumps.wikimedia.org/bewikibooks/latest/bewikibooks-latest-pages-articles-multistream.xml.bz2|wikimedia/bewikibooks-latest-pages-articles-multistream.xml.bz2|CC_BY-SA_GFDL"
  "bewikiquote_articles|Wikimedia|bewikiquote|https://dumps.wikimedia.org/bewikiquote/latest/bewikiquote-latest-pages-articles-multistream.xml.bz2|wikimedia/bewikiquote-latest-pages-articles-multistream.xml.bz2|CC_BY-SA_GFDL"
  "tatoeba_sentences|Tatoeba|tatoeba|https://downloads.tatoeba.org/exports/sentences.tar.bz2|tatoeba/sentences.tar.bz2|per-sentence_attribution"
  "tatoeba_links|Tatoeba|tatoeba_links|https://downloads.tatoeba.org/exports/links.tar.bz2|tatoeba/links.tar.bz2|per-sentence_attribution"
  "ud_belarusian_hse|UD|ud_hse|https://github.com/UniversalDependencies/UD_Belarusian-HSE/archive/refs/heads/master.zip|ud_belarusian_hse/master.zip|CC_BY-SA_4.0"
  "belacorpus_public|Belacorpus|belacorpus|https://github.com/Belarusian-Corpus/belacorpus_public/archive/refs/heads/main.zip|belacorpus_public/main.zip|manual_request_required"
)

ok=0
fail=0
failed_list=()
> "$STATUS_FILE"
> "$SHA256_FILE"

for src in "${sources[@]}"; do
  IFS='|' read -r source_id source_group source_name url rel_path license <<< "$src"
  out="$DOWNLOAD_DIR/$rel_path"
  out_dir="$(dirname "$out")"
  mkdir -p "$out_dir"

  echo ""
  echo "-------------------------------------------------"
  echo "START_DOWNLOAD source_id=$source_id"
  echo "  GROUP=$source_group"
  echo "  URL=$url"
  echo "  OUT=$out"
  echo "  LICENSE=$license"
  echo "-------------------------------------------------"

  if [[ -f "$out" ]] && [[ -s "$out" ]]; then
    sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    sha=$(sha256sum "$out" | awk '{print $1}')
    echo "SKIP_DOWNLOAD_ALREADY_EXISTS source_id=$source_id size=$sz sha256=$sha"
    echo "{\"source_id\":\"$source_id\",\"status\":\"skip_exists\",\"size\":$sz,\"sha256\":\"$sha\",\"timestamp\":\"$TS\"}" >> "$STATUS_FILE"
    echo "$sha  $out" >> "$SHA256_FILE"
    ok=$((ok + 1))
    continue
  fi

  # Try download
  http_code=0
  if curl \
      --location \
      --fail \
      --show-error \
      --progress-bar \
      --connect-timeout 30 \
      --max-time 1800 \
      --retry 5 \
      --retry-delay 10 \
      --continue-at - \
      -o "$out.part" \
      "$url" 2>&1; then
    mv "$out.part" "$out"
    sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    sha=$(sha256sum "$out" | awk '{print $1}')
    echo "DOWNLOAD_OK source_id=$source_id size=$sz sha256=$sha"
    echo "{\"source_id\":\"$source_id\",\"status\":\"ok\",\"size\":$sz,\"sha256\":\"$sha\",\"timestamp\":\"$TS\"}" >> "$STATUS_FILE"
    echo "$sha  $out" >> "$SHA256_FILE"
    ok=$((ok + 1))
  else
    rc=$?
    rm -f "$out.part"
    echo "DOWNLOAD_FAIL source_id=$source_id exit_code=$rc"
    echo "{\"source_id\":\"$source_id\",\"status\":\"fail\",\"exit_code\":$rc,\"timestamp\":\"$TS\"}" >> "$STATUS_FILE"
    fail=$((fail + 1))
    failed_list+=("$source_id")
  fi
done

ln -sf "download_v2_sha256_${TS}.txt" "$REPORT_DIR/downloads/download_v2_sha256_latest.txt" 2>/dev/null || true

echo ""
echo "============================================="
echo " DOWNLOAD SUMMARY"
echo "============================================="
echo "DOWNLOAD_DONE=$([ $fail -eq 0 ] && echo YES || echo NO)"
echo "SOURCES_OK=$ok"
echo "SOURCES_FAILED=$fail"
echo "FAILED_SOURCES=${failed_list[*]:-none}"
echo "STATUS_FILE=$STATUS_FILE"
echo "SHA256_FILE=$SHA256_FILE"
echo "LOG=$LOG"
echo ""
if [ $fail -eq 0 ]; then
  echo "NEXT_FOR_USER=\"bash dist/owner_runs/02_OWNER_BUILD_DATASET_V2.sh\""
else
  echo "Some sources failed. Check $STATUS_FILE for details."
  echo "You can re-run this script; already-downloaded sources will be skipped."
fi
echo "============================================="
