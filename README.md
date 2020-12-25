## YUVAccelerator

This is Lab 4.1 of PKU Computer Organization and Architecture. Do image processing acceleration based on SIMD instruction set extensions.

### Part 1

Implement gradual change of images. Process a single-frame yuv file. Convert it to rgb file, multiply different values of alpha, and then convert & save it to a same yuv file.

### Part 2

Implement mergence of two images, which looks like an image overlaying another one gradually.

### Build

To compile, run this command: (-march option is indispensable, otherwise AVX is not supported)

`g++ -march=native -o yuv yuv.cpp`

To run:

`./yuv 1 (or 2)`

1 or 2 indicates which part to run.

By default it doesn't save yuv file. You could add define at first if you need to see the yuv file result after processing acceleration. For example, add '#define avx' at first and run './yuv 1', and you' ll see 'fade_avx.yuv' file. But time elapsed of avx will be changed greatly because saving file takes much time.