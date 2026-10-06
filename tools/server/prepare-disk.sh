#!/usr/bin/env bash
# Prepare the second disk for the corpus and stop the atime writeback storm on the system disk.
#
# usage: sudo tools/server/prepare-disk.sh [/dev/sdb] [/mnt/corpus]
#   Formats the WHOLE device as ext4 (no partition table), mounts it noatime, adds an fstab entry by UUID, and remounts /
#   with noatime. Refuses to run if the device already has a filesystem or partitions.
#
# Why noatime: the first read of every tar-extracted file updates its atime, and with ~20M cached inodes the kernel
# writeback spins kworkers at 100% (soft lockups in dmesg); cold reads drop to ~1 MB/s. ml-train reads with O_NOATIME,
# but every other tool (grep, rsync, du) hits it.
set -euo pipefail
dev="${1:-/dev/sdb}"; mnt="${2:-/mnt/corpus}"
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
[ -b "$dev" ] || { echo "$dev is not a block device" >&2; exit 1; }
if [ -n "$(lsblk -no FSTYPE "$dev" | tr -d '[:space:]')" ] || [ "$(lsblk -no NAME "$dev" | wc -l)" -gt 1 ]; then
  echo "$dev already has a filesystem or partitions — refusing to format" >&2; lsblk -f "$dev" >&2; exit 1
fi
if mountpoint -q "$mnt"; then echo "$mnt is already mounted" >&2; exit 1; fi

echo "formatting $dev as ext4 (label corpus)"
mkfs.ext4 -q -L corpus "$dev"
mkdir -p "$mnt"
mount -o noatime "$dev" "$mnt"

uuid="$(blkid -s UUID -o value "$dev")"
if ! grep -q "UUID=$uuid" /etc/fstab; then
  echo "UUID=$uuid $mnt ext4 noatime,nofail 0 2" >> /etc/fstab
fi

# system disk: noatime now and after reboot
mount -o remount,noatime /
if ! awk '$2=="/" && $4 ~ /noatime/ {found=1} END {exit !found}' /etc/fstab; then
  cp /etc/fstab /etc/fstab.bak
  awk '$2=="/" && $4 !~ /noatime/ {$4="noatime,"$4} {print}' /etc/fstab > /etc/fstab.new && mv /etc/fstab.new /etc/fstab
fi

mount -a
echo; df -h "$mnt" /; echo; grep -E " (/|$mnt) " /etc/fstab
