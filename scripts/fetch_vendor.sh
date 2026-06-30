#!/usr/bin/env bash
# Download and verify etcd + nats-server binaries into vendor/bin/.
# Run once after cloning the repo (requires internet access from the login node).
# Verifies SHA256 of each extracted binary against values in config/versions.env.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

VENDOR="$REPO_ROOT/vendor/bin"
TMP=$(mktemp -d /tmp/fetch-vendor-XXXX)
trap "rm -rf $TMP" EXIT

mkdir -p "$VENDOR"

# ── etcd ─────────────────────────────────────────────────────────────────────
ETCD_ARCHIVE="etcd-v${ETCD_VERSION}-linux-amd64.tar.gz"
ETCD_URL="https://github.com/etcd-io/etcd/releases/download/v${ETCD_VERSION}/${ETCD_ARCHIVE}"

if [ -f "$VENDOR/etcd" ]; then
    ACTUAL=$(sha256sum "$VENDOR/etcd" | awk '{print $1}')
    if [ "$ACTUAL" = "$ETCD_SHA256" ]; then
        echo "etcd v${ETCD_VERSION}: already present and verified ✅"
    else
        echo "etcd: checksum mismatch — re-downloading"
        rm -f "$VENDOR/etcd"
    fi
fi

if [ ! -f "$VENDOR/etcd" ]; then
    echo "Downloading etcd v${ETCD_VERSION}..."
    curl -fL --progress-bar "$ETCD_URL" -o "$TMP/$ETCD_ARCHIVE"
    tar -xzf "$TMP/$ETCD_ARCHIVE" -C "$TMP" --strip-components=1 \
        "etcd-v${ETCD_VERSION}-linux-amd64/etcd"
    ACTUAL=$(sha256sum "$TMP/etcd" | awk '{print $1}')
    if [ "$ACTUAL" != "$ETCD_SHA256" ]; then
        echo "ERROR: etcd SHA256 mismatch!"
        echo "  expected: $ETCD_SHA256"
        echo "  got:      $ACTUAL"
        exit 1
    fi
    cp "$TMP/etcd" "$VENDOR/etcd"
    chmod +x "$VENDOR/etcd"
    echo "etcd v${ETCD_VERSION}: downloaded and verified ✅"
fi

# ── nats-server ───────────────────────────────────────────────────────────────
# Note: nats-server switched from .zip to .tar.gz in v2.11+
NATS_ARCHIVE="nats-server-v${NATS_VERSION}-linux-amd64.tar.gz"
NATS_URL="https://github.com/nats-io/nats-server/releases/download/v${NATS_VERSION}/${NATS_ARCHIVE}"

if [ -f "$VENDOR/nats-server" ]; then
    ACTUAL=$(sha256sum "$VENDOR/nats-server" | awk '{print $1}')
    if [ "$ACTUAL" = "$NATS_SHA256" ]; then
        echo "nats-server v${NATS_VERSION}: already present and verified ✅"
    else
        echo "nats-server: checksum mismatch — re-downloading"
        rm -f "$VENDOR/nats-server"
    fi
fi

if [ ! -f "$VENDOR/nats-server" ]; then
    echo "Downloading nats-server v${NATS_VERSION}..."
    curl -fL --progress-bar "$NATS_URL" -o "$TMP/$NATS_ARCHIVE"
    tar -xzf "$TMP/$NATS_ARCHIVE" -C "$TMP" \
        "nats-server-v${NATS_VERSION}-linux-amd64/nats-server"
    ACTUAL=$(sha256sum "$TMP/nats-server-v${NATS_VERSION}-linux-amd64/nats-server" | awk '{print $1}')
    if [ "$ACTUAL" != "$NATS_SHA256" ]; then
        echo "ERROR: nats-server SHA256 mismatch!"
        echo "  expected: $NATS_SHA256"
        echo "  got:      $ACTUAL"
        exit 1
    fi
    cp "$TMP/nats-server-v${NATS_VERSION}-linux-amd64/nats-server" "$VENDOR/nats-server"
    chmod +x "$VENDOR/nats-server"
    echo "nats-server v${NATS_VERSION}: downloaded and verified ✅"
fi

echo ""
echo "vendor/bin ready:"
ls -lh "$VENDOR"
