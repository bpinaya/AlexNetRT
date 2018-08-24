#!/bin/sh
FILENAME=$(basename -- $1)
FILENAME="${FILENAME%.*}"

convert $1 -resize 227x227 -background white -gravity center -extent 227x227 $FILENAME.ppm