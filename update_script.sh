#!/bin/bash

# Faylda avtomatik dəyişiklik etmək
echo "Auto update at $(date)" >> auto_update_log.txt

# Git commit əmrləri
git add .
git commit -m "Auto commit: $(date)"
