# Lipsync

## How it works

### 1. Using TensorFlow.js facemesh

The TensorFlow facemesh model provides real-time high density estimate of key points of your facial expression using only a webcam and on device machine learning. We use the key points around the mouth and lips to estimate how well you synchronize to the lyrics of the song.

> More about TensorFlow.js facemesh: https://github.com/tensorflow/tfjs-models/tree/master/facemesh



### 2. Measuring the shape of the mouth

What is a mouth shape? There are many different ways to measure the shape of your mouth. We needed a technique that allows the user to move their head around while singing and is relatively forgiving in different mouth shapes, sizes, and distance to the camera.

We decided to use OpenCV matchShapes Hu Moments. In the OpenCV library, there is a matchShapes function which compares contours and returns a similarity score. Underneath the hood, the matchShapes function uses a technique called Hu Moments which provides a set of numbers calculated using central moments that are invariant to image transformations. This allowed us to compare shapes regardless of translation, scale, and rotation. So the user can freely rotate their head without impacting the detection of the mouth shape itself.

In order to use OpenCV matchShapes to compare the mouth shape, we would need to create an image from the facial key points. So, we use the key points around the mouth to create a black and white image for both the baseline and the user input. 


## Setup

Use serve (https://github.com/vercel/serve#readme) to serve static file in directory by running:
``` shell
serve
```

## Live Demo

You can checkout a full live demo of this in action here: https://lipsync.withyoutube.com
