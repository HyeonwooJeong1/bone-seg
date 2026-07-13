#!/usr/bin/env bash
# Download status snapshot (GPU-free). Usage: bash /data1/bone/ai_bone/dl_status.sh
RAW=/data1/bone/raw
LOGS=/data1/bone/dl_logs
echo "===== $(date '+%F %T') ====="
echo "-- running downloaders --"
pgrep -af "ai_bone.download|snapshot_download|huggingface" | grep -v pgrep || echo "  (none running)"
echo "-- sizes ($RAW) --"
du -sh "$RAW"/* 2>/dev/null | sort -k2 || echo "  (no data yet)"
echo "-- disk (/data1) --"
df -h /data1 | tail -1
echo "-- last log line each --"
for f in "$LOGS"/*.log; do [ -e "$f" ] || continue; printf '  %-16s ' "$(basename "$f")"; tail -n 1 "$f" 2>/dev/null; done
