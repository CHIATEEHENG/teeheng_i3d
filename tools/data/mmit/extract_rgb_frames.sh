#! /usr/bin/bash env

cd ../
python build_rawframes.py ../../data/mmit/videos/ ../../data/mmit/rawframes/ --task rgb --level 2 --ext mp4

echo "Genearte raw frames (RGB only)"

cd mmit/
