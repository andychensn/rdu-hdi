#!/usr/bin/env bash
for p in /proc/[0-9]*; do
    pid=$(basename "$p")
    if ls -la "$p/fd" 2>/dev/null | grep -q "/dev/rdu"; then
        echo "PID $pid holds /dev/rdu*:"
        ps -f -p "$pid" 2>/dev/null | tail -1
    fi
done
echo "=== check complete ==="
