#!/bin/bash
# nfsim.sh — NFSv4 import /data/tensordata from DGX on PGX
#
# Usage (on PGX as root, or via sudo):
#   sudo ./nfsim.sh             # mount + verify
#   sudo ./nfsim.sh --undo      # umount
#   sudo ./nfsim.sh --test      # dd-speedtest auf ein existierendes Modell-Shard
#
# Mounts DGX:/data/tensordata (read-only, NFSv4) to /data/tensordata-dgx over
# the 200 Gb/s direct link (192.168.0.117).

set -euo pipefail

SERVER_IP="192.168.0.117"
REMOTE_PATH="/data/tensordata"
LOCAL_MOUNT="/data/tensordata-dgx"
MOUNT_OPTS="ro,hard,rsize=1048576,wsize=1048576,proto=tcp"

if [[ $EUID -ne 0 ]]; then
    echo "Muss als root laufen: sudo $0 $*"
    exit 1
fi

already_mounted() {
    mountpoint -q "$LOCAL_MOUNT"
}

if [[ "${1:-}" == "--undo" ]]; then
    if already_mounted; then
        echo "umount $LOCAL_MOUNT"
        umount "$LOCAL_MOUNT" || {
            echo "Lazy umount als Fallback..."
            umount -l "$LOCAL_MOUNT"
        }
    else
        echo "$LOCAL_MOUNT ist nicht gemountet."
    fi
    exit 0
fi

if [[ "${1:-}" == "--test" ]]; then
    if ! already_mounted; then
        echo "FEHLER: $LOCAL_MOUNT nicht gemountet. Zuerst: sudo $0"
        exit 1
    fi
    # Pick a safetensor to dd over
    TEST_FILE=$(find "$LOCAL_MOUNT" -maxdepth 3 -name "*.safetensors" 2>/dev/null | head -1)
    if [[ -z "$TEST_FILE" ]]; then
        echo "Keine .safetensors-Datei unter $LOCAL_MOUNT gefunden."
        exit 1
    fi
    SIZE=$(stat -c %s "$TEST_FILE")
    SIZE_GB=$(awk -v s="$SIZE" 'BEGIN { printf "%.2f", s/1024/1024/1024 }')
    echo "Speedtest: $TEST_FILE ($SIZE_GB GB)"
    echo "dd if=... of=/dev/null bs=1M"
    dd if="$TEST_FILE" of=/dev/null bs=1M status=progress 2>&1 | tail -3
    exit 0
fi

# Sanity
mkdir -p "$LOCAL_MOUNT"

if already_mounted; then
    echo "Bereits gemountet:"
    mount | grep "$LOCAL_MOUNT"
    echo ""
    echo "Neu mounten? Zuerst: sudo $0 --undo"
    exit 0
fi

echo "Mounting $SERVER_IP:$REMOTE_PATH → $LOCAL_MOUNT (NFSv4, $MOUNT_OPTS)"
mount -t nfs4 -o "$MOUNT_OPTS" "$SERVER_IP:$REMOTE_PATH" "$LOCAL_MOUNT"

echo ""
echo "OK. Inhalt:"
ls "$LOCAL_MOUNT" | head -10

echo ""
echo "Speedtest laufen lassen: sudo $0 --test"
