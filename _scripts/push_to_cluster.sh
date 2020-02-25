#!/usr/bin/env bash

# For Ubuntu 18.04, also add the lang specs to the .bashrc!
# export LC_CTYPE=en_US.UTF-8

#rsync -rPz -e 'ssh -p 2223' --exclude '*.png' --exclude 'data/' --exclude '*.tsv' --exclude 'venv/' --exclude 'analysis/' --exclude '.env' --progress /Users/david/GoogleDrive/_MasterThesis/ david@77.59.149.134:/home/david/_MasterThesis/
rsync -rPz -e 'ssh -p 2223' --exclude '*.png' --exclude '*.tsv' --exclude 'venv/' --exclude 'analysis/' --exclude '.env' --progress /Users/david/GoogleDrive/_MasterThesis/ david@77.59.149.134:/home/david/_MasterThesis/
