#!/bin/bash
# nfsex.sh — NFSv4 export /data/tensordata from DGX to PGX (direct link only)
#
# Usage (on DGX as root, or via sudo):
#   sudo ./nfsex.sh           # add + reload exports
#   sudo ./nfsex.sh --undo    # remove export entry + reload
#
# Restricts access to 192.168.0.116 (PGX on enp1s0f0np0 200 Gb/s direct link)
# Read-only — pack-time BF16 read only.

set -euo pipefail

EXPORT_PATH="/data/tensordata"
CLIENT_IP="192.168.0.116"
EXPORT_OPTS="ro,sync,no_subtree_check,no_root_squash"
EXPORTS_LINE="$EXPORT_PATH $CLIENT_IP($EXPORT_OPTS)"
EXPORTS_FILE="/etc/exports"

if [[ $EUID -ne 0 ]]; then
    echo "Muss als root laufen: sudo $0 $*"
    exit 1
fi

if [[ "${1:-}" == "--undo" ]]; then
    if grep -qF "$EXPORTS_LINE" "$EXPORTS_FILE"; then
        echo "Entferne: $EXPORTS_LINE"
        grep -vF "$EXPORTS_LINE" "$EXPORTS_FILE" > "$EXPORTS_FILE.tmp"
        mv "$EXPORTS_FILE.tmp" "$EXPORTS_FILE"
        exportfs -ra
        echo "Aktive Exports:"
        exportfs -v
    else
        echo "Eintrag nicht gefunden, nichts zu tun."
    fi
    exit 0
fi

# Sanity
if [[ ! -d "$EXPORT_PATH" ]]; then
    echo "FEHLER: $EXPORT_PATH existiert nicht."
    exit 1
fi

# Idempotent: nur hinzufügen wenn nicht schon da
if grep -qF "$EXPORTS_LINE" "$EXPORTS_FILE"; then
    echo "Eintrag bereits in $EXPORTS_FILE vorhanden."
else
    echo "Füge Eintrag hinzu: $EXPORTS_LINE"
    echo "$EXPORTS_LINE" >> "$EXPORTS_FILE"
fi

# NFS server + exports reload
systemctl is-active --quiet nfs-server || systemctl start nfs-server
exportfs -ra

echo ""
echo "Aktive Exports:"
exportfs -v

echo ""
echo "Test von PGX aus:"
echo "  ssh flash@192.168.0.116 'sudo mount -t nfs4 -o ro,hard,rsize=1048576,wsize=1048576,proto=tcp 192.168.0.117:/data/tensordata /data/tensordata-dgx'"
