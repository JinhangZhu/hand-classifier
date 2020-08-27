#!/bin/bash

# Download zip dataset from Google Drive
filename='handcnn-gpu.pt'
fileid='1eXbnqNdKxBYc8p0G8RClO376HW5kN2vR'
curl -L -o ${filename} "https://drive.google.com/uc?export=download&id=${fileid}"
echo "Weight file ${filename} saved."

filename='handcnn-cpu.pt'
fileid='1lWAehfLX37MO6fA6bbKqxLhBMT3Y-BT6'
curl -L -o ${filename} "https://drive.google.com/uc?export=download&id=${fileid}"
echo "Weight file ${filename} saved."
