#!/bin/bash

mkdir datasets
cd datasets

# Download zip dataset from Google Drive
filename='handcrops.zip'
fileid='1t0d9lRkzL4Qv1NdWbYsZbc9lQYF9MS9W'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" >/dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileid}" -o ${filename}
rm ./cookie

# Unzip
unzip -q ${filename}
rm ${filename}

cd