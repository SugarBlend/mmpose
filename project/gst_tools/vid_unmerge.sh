#!/bin/bash

FILEPATH="$1"
W="$2"
H="$3"
OUTDIR="$4"
PREFIX="$5"

mkdir -p "${OUTDIR}"

gst-launch-1.0 filesrc location="${FILEPATH}" ! decodebin ! videoconvert ! tee name=t \
t. ! queue ! videocrop top=0 left=0 right=${W} bottom=${H} ! x264enc ! mp4mux ! filesink location="${OUTDIR}/cam0${RREFIX}.mp4" \
t. ! queue ! videocrop top=0 left=${W} right=0 bottom=${H} ! x264enc ! mp4mux ! filesink location="${OUTDIR}/cam1${PREFIX}.mp4" \
t. ! queue ! videocrop top=${H} left=0 right=${W} bottom=0 ! x264enc ! mp4mux ! filesink location="${OUTDIR}/cam2${PREFIX}.mp4" \
t. ! queue ! videocrop top=${H} left=${W} right=0 bottom=0 ! x264enc ! mp4mux ! filesink location="${OUTDIR}/cam3${PREFIX}.mp4"
