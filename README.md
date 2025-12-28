---
title: Doodle Recognition
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Doodle Recognition

A deep learning web app that recognizes hand-drawn doodles in real-time.

## Features

- **Real-time prediction** as you draw
- **Multiple models**: ResNet classification and OpenCV similarity matching
- **Challenge mode**: Get random words to draw and earn points
- **340 categories** of doodles to recognize

## Models

- **ResNet**: Deep learning classification model trained on Quick, Draw! dataset
- **Similarity**: OpenCV-based template matching using SIFT features

## Tech Stack

- Flask web server
- PyTorch for deep learning
- OpenCV for similarity matching
- HTML5 Canvas for drawing
