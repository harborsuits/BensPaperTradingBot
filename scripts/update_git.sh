#!/bin/bash
# Simple script to update git repository

cd "/Users/bendickinson/Desktop/Trading:BenBot"
git add .
git commit -m "Update frontend API integration with backend"
git push
echo "Git repository updated successfully!"
