Note: You need to be able to run a python code to use this tool!

-the wavetable which will be processed needs to be in the same folder as the .py file and needs to have the following name: "wavetable_input.wav"
-the code does the convolution product between the wavetable and a kernel. 
There are 3 available kernel:
kernel 1: the pixel value at the position (x,y) in the output wavetable is the average of the pixels in a square around the position (x,y) 
kernel 2: the pixel value at the position (x,y) in the output wavetable is the average of the pixels in a square around the position (x,y) , weighted with a bell curve. This is a Gaussian Blur
kernel 3: edge detection
-feel free to create your own kernels
-make to have  sample_number=2048 if you use serum or 1024 if you use ableton wavetable
-the output wavetable will have the name 'wavetable_result.wav', and will be in the same folder as the .py file
-to import the wavetable in Serum: go in the wavetable editor and write 2048 in the formula parser. Then drag and drop thewav file in Serum.
-to import the wavetable in Ableton Wavetable: Click on the "raw" button. Then drag and drop the wav file in Ableton Wavetable.
