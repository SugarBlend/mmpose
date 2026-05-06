#!/bin/bash

FILEPATH="$1"
W="$2"
H="$3"

NAME = $(basename "${FILEPATH}")
NAME = "${NAME%.*}"
DIRPATH = $(dirname "${FILEPATH}")
OUTDIR = "${DIRPATH}/${NAME}"

rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}/TL"
mkdir -p "${OUTDIR}/TR"
mkdir -p "${OUTDIR}/BL"
mkdir -p "${OUTDIR}/BR"

gst-launch-1.0 filesrc location="${FILEPATH}" ! decodebin ! videoconvert ! tee name=t \
t. ! queue ! videocrop top=0 left=0 right=${W} bottom=${H} ! videorate ! jpegenc ! multifilesink location="${OUTDIR}/TL/%06d.jpg" \
t. ! queue ! videocrop top=0 left=${W} right=0 bottom=${H} ! videorate ! jpegenc ! multifilesink location="${OUTDIR}/TR/%06d.jpg" \
t. ! queue ! videocrop top=${H} left=0 right=${W} bottom=0 ! videorate ! jpegenc ! multifilesink location="${OUTDIR}/BL/%06d.jpg" \
t. ! queue ! videocrop top=${H} left=${W} right=0 bottom=0 ! videorate ! jpegenc ! multifilesink location="${OUTDIR}/BR/%06d.jpg"
