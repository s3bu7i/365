#!/bin/bash

# Repo-nun yerli yolunu daxil edin
REPO_PATH="https://github.com/s3bu7i/365"
cd $REPO_PATH

# Faylı yenilə (random mətn əlavə et)
echo "Update: $(date)" >> updates.txt

# Git əmrləri
git add updates.txt
git commit -m "Auto commit at $(date)"
git push origin main
