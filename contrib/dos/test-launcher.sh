#!/bin/sh
# Run xpar under DOSBox-X while translating only the harness's file view.

set -e
: "${DOSBOX_REQUEST_ROOT:?DOSBOX_REQUEST_ROOT is required}"
: "${DOSBOX_BRIDGE:?DOSBOX_BRIDGE is required}"
: "${DOSBOX_EXEC:?DOSBOX_EXEC is required}"

key=`printf '%s' "$PWD" | cksum | awk '{print $1 "-" $2}'`
state=$DOSBOX_REQUEST_ROOT/names.$key
DOSBOX_NAME_STATE=$state; export DOSBOX_NAME_STATE
sh "$DOSBOX_BRIDGE" pre "$state" "$@"
status=0
"$DOSBOX_EXEC" XPAR.EXE "$@" || status=$?
sh "$DOSBOX_BRIDGE" post "$state" "$@"
exit "$status"
