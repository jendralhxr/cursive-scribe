#!/bin/bash
#
# Downloads all 255 images for manuscript DS 0097 00014 from vhmml.org
# using the IIIF Image API full-resolution endpoint.
#
# Usage:
#   chmod +x download_DS_0097_00014.sh
#   ./download_DS_0097_00014.sh

set -uo pipefail

BASE="https://www.vhmml.org/image/READING_ROOM/DS/DS%200097%2000014"
OUTDIR="DS_0097_00014_images"
UA="Mozilla/5.0"
IIIF_SUFFIX="/full/full/0/default.jpg"
EXT="JPG"

mkdir -p "$OUTDIR"

# Build the list of folio labels: 001r,001v,002r,002v,...,124r,124v,
# then bc,bp,fc,fp,x01,x02,x03
labels=()
for i in $(seq -w 1 124); do
    labels+=("${i}r")
    labels+=("${i}v")
done
labels+=("bc" "bp" "fc" "fp" "x01" "x02" "x03")

total=${#labels[@]}
count=0
failed=()

for label in "${labels[@]}"; do
    count=$((count + 1))
    fname="DS_0097_00014_${label}.${EXT}"
    url="${BASE}//${fname}${IIIF_SUFFIX}"
    outfile="${OUTDIR}/${fname}"

    if [ -f "$outfile" ]; then
        echo "[$count/$total] Skipping $fname (already exists)"
        continue
    fi

    echo "[$count/$total] Downloading $fname"

    http_code=$(curl -s -A "$UA" -o "$outfile" -w "%{http_code}" "$url")

    if [ "$http_code" != "200" ]; then
        echo "  -> FAILED (HTTP $http_code)"
        failed+=("$fname")
        rm -f "$outfile"
    fi

    # Be polite to the server
    sleep 0.5
done

echo ""
echo "Done. $((total - ${#failed[@]}))/$total downloaded successfully."

if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed downloads (${#failed[@]}):"
    printf '  %s\n' "${failed[@]}"
fi
