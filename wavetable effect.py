import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.io import wavfile


sample_number=2048
waveform_number=256

total_number_of_samples=sample_number*waveform_number

print(total_number_of_samples)

samplerate, data = wavfile.read('wavetable_input.wav')

print(data.shape)

wavetable=np.zeros([waveform_number,sample_number])

for i in range(waveform_number):
    for j in range(sample_number):
        wavetable[i,j]=data[i*sample_number+j]


fig = plt.figure()
plt.imshow(wavetable)
plt.show() 

#choose a square kernel with  odd size

kernel_number=1

if kernel_number==1:
    kernel_size=21
    kernel=np.ones([kernel_size,kernel_size])
    
    halfsize=int((kernel_size-1)/2)
    kernel_nb_elements=kernel_size**2

if kernel_number==2:
    kernel_size=101
    x = np.linspace(-1, 1, kernel_size)
    y = np.linspace(-1, 1, kernel_size)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(2*X)**2-(2*Y)**2)
    
    halfsize=int((kernel_size-1)/2)
    kernel_nb_elements=kernel_size**2

if kernel_number==3:
    kernel_size=11
    kernel=np.zeros([ kernel_size,  kernel_size])
    
    halfsize=int((kernel_size-1)/2)
    kernel_nb_elements=kernel_size**2
    
    kernel[:,0: halfsize]=-1
    kernel[:, halfsize+1: kernel_size]=1
    


fig = plt.figure()
plt.imshow(kernel)
plt.show() 






size_added_to_one_side=100
wavetable_filtered=np.zeros([waveform_number,sample_number])
large_wavetable=np.ones([waveform_number+size_added_to_one_side*2,sample_number+size_added_to_one_side*2])

large_wavetable[size_added_to_one_side:size_added_to_one_side+waveform_number,size_added_to_one_side:size_added_to_one_side+sample_number]=wavetable[:,:]


# fig = plt.figure()
# plt.imshow(large_wavetable)
# plt.show() 

for i in range(size_added_to_one_side,size_added_to_one_side+waveform_number):
    for j in range(0, size_added_to_one_side):
        large_wavetable[i,j]=wavetable[i-size_added_to_one_side,0]
        
for i in range(size_added_to_one_side,size_added_to_one_side+waveform_number):
    for j in range(sample_number+size_added_to_one_side,sample_number+2*size_added_to_one_side):
        large_wavetable[i,j]=wavetable[i-size_added_to_one_side,sample_number-1]
        
for i in range(0,size_added_to_one_side):
    for j in range(size_added_to_one_side,size_added_to_one_side+sample_number):
        large_wavetable[i,j]=wavetable[0,j-size_added_to_one_side]
        
for i in range(size_added_to_one_side+waveform_number,size_added_to_one_side+waveform_number+size_added_to_one_side):
    for j in range(size_added_to_one_side,size_added_to_one_side+sample_number):
        large_wavetable[i,j]=wavetable[waveform_number-1,j-size_added_to_one_side]
        

def multiply_and_sum(large_wavetable,kernel,i,j):
    function_multiplied_by_kernel=np.multiply(large_wavetable[i+size_added_to_one_side-halfsize:i+size_added_to_one_side+halfsize+1, j+size_added_to_one_side-halfsize:j+size_added_to_one_side+halfsize+1],kernel)
    return np.sum(function_multiplied_by_kernel)/kernel_nb_elements


fig = plt.figure()
plt.imshow(large_wavetable)
plt.show() 

for i in range(waveform_number):
    for j in range(sample_number):
       wavetable_filtered[i,j]=multiply_and_sum(large_wavetable,kernel,i,j)
        
        
fig = plt.figure() 
plt.imshow(wavetable_filtered)
plt.show() 
        

#the .wav is created
wav=np.ones(sample_number*waveform_number, dtype='float32') #dtype='float32' is import for ableton to read the files.
#but serum is ok with reading 64 bit files so it's not necessary to use the command for serum

for i in range(waveform_number):
    for j in range(sample_number):
        wav[j+i*sample_number]=wavetable_filtered[i][j]

rate = 44100 # any number can be used

write('wavetable_result.wav', rate, wav)

        
        
