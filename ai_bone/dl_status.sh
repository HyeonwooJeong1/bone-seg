#!/usr/bin/env bash
# Clean download status table (GPU-free). Usage: bash /data1/bone/ai_bone/dl_status.sh
RAW=/data1/bone/raw
LOGS=/data1/bone/dl_logs
echo "===== downloads  $(date '+%F %T') ====="
printf '  %-14s %8s  %-6s %9s\n' DATASET SIZE STATUS FILES
if compgen -G "$RAW/*/" >/dev/null 2>&1; then
  for d in "$RAW"/*/; do
    ds=$(basename "$d")
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    files=$(find "$d" -type f 2>/dev/null | wc -l)
    if pgrep -f "ai_bone.download $ds" >/dev/null 2>&1; then st="RUN"; else st="idle"; fi
    printf '  %-14s %8s  %-6s %9s\n' "$ds" "$size" "$st" "$files"
  done
else
  echo "  (no downloads yet)"
fi
echo "-- disk (/data1) --"
df -h --output=used,avail,pcent /data1 2>/dev/null | tail -1 \
  | awk '{print "  "$1" used, "$2" free ("$3")"}'
cadsp=$(grep -ao 'Fetching[^0-9]*[0-9]*it' "$LOGS/cads.log" 2>/dev/null | tail -1)
[ -n "$cadsp" ] && echo "-- CADS $cadsp"
# any recent errors across logs (429/403/traceback)?
errs=$(grep -lE "403|Traceback|GatedRepo|No such file" "$LOGS"/*.log 2>/dev/null | xargs -r -n1 basename | paste -sd, -)
[ -n "$errs" ] && echo "-- ! check logs with errors: $errs"
exit 0
