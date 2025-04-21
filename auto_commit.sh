#!/bin/bash
set -e

# Faylı yenilə (random mətn əlavə et)
echo "Update: $(date)" >> updates.txt

# Git əmrləri
git add updates.txt
git commit -m "Auto commit at $(date)"
git push origin main

