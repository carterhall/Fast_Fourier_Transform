# Fast\_Fourier\_Transform

A fast, lightweight FFT written in [Jai](https://github.com/Jai-Community/Jai-Community-Library/wiki), intended for realtime audio applications. MIT licensed.

This repo is structured as a Jai module; you should be able to just clone it into your modules directory.

## Example (simple, global API)
  
    FFT :: #import "Fast_Fourier_Transform"(512);  // At compiletime, prepare FFTs up to size 512

    main :: () {
      N :: 64;
      signal           := NewArray(N, float32);  // Real-valued signal (it could also be complex)
      spectrum         := NewArray(N, Complex);  // The spectrum is always complex
      recovered_signal := NewArray(N, float32);

      FFT.forward(signal, spectrum);  // Transform real signal to complex spectrum

      FFT.inverse(spectrum, recovered_signal);  // Recover the real signal from the spectrum
    }

    #import "Math";
    #import "Basic";

## Example (standard API)

    FFT :: #import "Fast_Fourier_Transform";  // Don't initialize the global FFT at compiletime

    main :: () {
      N :: 64;
      
      fft_data := FFT.prepare_fft(N);  // Creates an FFT_Data struct
      defer FFT.free_fft(*fft_data);
 
      // In the global API example above, we simply use a global FFT_Data struct, and call
      // prepare_fft() at compiletime to initialize it.
      // That's very convenient, but it requires us to know the maximum size transform the program
      // will ever need to do. If you need to allow arbitrarily large transforms, use this version.

      signal           := NewArray(N, Complex);  // We'll use a complex signal this time
      spectrum         := NewArray(N, Complex);
      recovered_signal := NewArray(N, Complex);

      FFT.forward(*fft_data, signal, spectrum);  // Transform complex signal to complex spectrum

      FFT.inverse(*fft_data, spectrum, recovered_signal);  // Recover the signal from the spectrum
    }

    #import "Math";
    #import "Basic";

## Usage and gotchas

### Gotcha 1: Input destruction

To avoid extra allocations, we use the input buffer to store intermediate calculations. If you will need your inputs again in the future, save them somewhere else. This also means **the input and output buffers cannot refer to the same memory.**

### Gotcha 2: Input/output buffer sizes must be powers of 2

The FFT is much simpler and easier to optimize if we restrict the buffers to power-of-2 sizes. If you need to transform an input of a different size, you can [zero-pad](https://www.bitweenie.com/listings/fft-zero-padding/) at the end.

### Real-valued signal size

When you call `prepare_fft(N)`, you are specifying the max transform size for *complex* signals. **You can use a complex transform of size N for a real-valued signal up to size 2N.** There's more info below on why this is possible. The internal FFT procedure will make sure you have called it with valid array sizes. 

### Memory footprint

An `FFT_Data` struct initialized for size N will take 32N bytes for the twiddle factors, and 8N bytes for the extra real-transform twiddles. This gives you the ability to do forward and inverse transforms for any size \[2,N] (complex) or \[2,2N] (real). If you know your program only needs to go in one direction, you can free the unneeded twiddles after calling `prepare_fft`, or modify that function to avoid creating them in the first place.

Note that if you use the global compiletime FFT, this memory footprint will be baked into the executable, which may become suboptimal at large sizes (for example, 327 KB for a complex transform size of 8192.)

## How it works

This module implements a radix-8 decimation-in-time FFT, with the Stockham auto-sort algorithm (all explained below.)

The base case, used for size-4 and below, is a naive DFT with hardcoded simple twiddle factors.

On modern CPUs, committing to radix-8 maximizes performance\*simplicity. The radix-8 butterfly can take great advantage of vector instructions, but is not so large that the implementation/unrolling becomes too unwieldy. The base-case naive DFT for size 4 and below can be easily optimized to avoid unnecessary multiplications, and is not large enough to suffer meaningfully from its quadratic complexity. The combination of radix-8 and hardcoded base cases for 4-and-down avoids the large number of recursive calls in more naive FFT implementations.

### The Cooley-Tukey algorithm

[Per Wikipedia](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm#Variations), the naive discrete Fourier transform is defined:

![DFT](https://wikimedia.org/api/rest_v1/media/math/render/svg/9b1598508ebb847e6c726d5b741ae2363d84f616)

The `O(N log(N))` FFT algorithm is made possible by re-indexing and rearranging this formula. We will split the input array into `N1` blocks of size `N2`, and define new indices `k1` and `n1` (each running from `0..N1-1`) and `k2` and `n2` (running from `0..N2-1`).

The output index `k` can now be rewritten as `N2*k1 + k2`, and the input index `n` can be rewritten `N1*n2 + n1`, in the complicated-looking formula

![Factorization](https://wikimedia.org/api/rest_v1/media/math/render/svg/36142c14b057685d73f85d9d15c7fd35f17cf1a7)

which, critically, can be re-expressed like:

![Cooley-Tukey factorization](https://wikimedia.org/api/rest_v1/media/math/render/svg/2a5a7489f7fcfc253bfe2a6a382103d60c027532)

In this last formula, the sum inside the parentheses is a DFT of size `N2`, and we can recursively call the FFT to calculate it, yielding the desired `N log(N)` complexity.

### Implementation

We fix `N1 = 8`, i.e. we choose a radix-8 decimation-in-time FFT. If we chose to fix `N2` instead, it would be called "decimation-in-frequency". 

We use the Stockham auto-sort algorithm, which entails passing intermediate values back and forth between the input and output buffers. This approach has good cache properties, and avoids reordering the output relative to the input, which some in-place FFT algorithms do. Spectrum reordering is not a problem for certain applications, like simple FFT convolution, but is harder to work with in other cases.

### Real-valued FFTs

For many applications, you will only ever want to perform FFTs on real-valued signals. It would be wasteful to do transforms on buffers that are half zeros, but luckily, it's not too hard to get 2 FFTs for the price of ~1 if the signal is known to be real. [This post](http://www.robinscheibler.org/2013/02/13/real-fft.html) explains it concisely. These optimizations are applied automatically if you call `forward` or `inverse` with a signal of type float32.

### Performance, SIMD, inline assembly

This module uses Jai's inline assembly support to implement fast versions of the required complex number operations for SSE, SSE3, AVX, and AVX-512. At runtime, the system's SIMD support will be detected automatically.

This module was benchmarked against [muFFT](https://github.com/Themaister/muFFT), a fast C implementation, and the two traded wins at various transform sizes (Ryzen 7700X). If you want to compare the two, you could compile muFFT, generate bindings, and call it in `test/test_fft.jai`. I did not include that functionality here, because it would have added many unnecessary files to the repo.

Currently, the AVX and AVX-512 implementations don't offer much benefit over the SSE3 implementation, and are even slightly worse in some cases. Hopefully this will be improved in the future. If you are interested in absolute maximum performance, you should run `test/test_fft.jai` and see what is fastest on your system. You can then manually set the module's global `simd_support` field after calling `prepare_fft`.

## Acknowledgements

This module was somewhat inspired by [muFFT](https://github.com/Themaister/muFFT), a C library that is almost as fast as FFTW, but much smaller and simpler.

And as always, thanks to Jonathan Blow for the language.

## Miscellany

### 1. Apple Silicon

At the time of writing, it is not yet easy to build Jai programs for arm64, and there is no support for inline assembly. This module still runs quite fast in SSE3 mode through Rosetta, but if you are from the future, and this module has not been updated with arm64 support, you should consider generating bindings for [vDSP](https://developer.apple.com/documentation/accelerate/vdsp?language=objc) and using Apple's FFT.

### 2. The FFT's bizarre history

Gauss invented the Fast Fourier Transform in 1805, wrote it in a notebook (in Latin), and refused to elaborate further. Then, in 1822, Fourier invented the regular Fourier Transform. (???) Finally, Cooley and Tukey re-invented the FFT in 1965, primarily to detect Soviet nuclear tests with faraway sensors.

