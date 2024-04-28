//
// This file tests forward transforms against a naive DFT, then tests inverse transforms by attempting to
// recover the input to the forward FFT. 
//
// Earlier tests just generated random data as input to each type of transform. This does not appear to be
// a valid approach to testing the inverse real-signal transform: different implementations will return
// different signals, if the input spectrum does not actually have the expected symmetric properties.
//
// I'm not sure exactly why this is the case, but it appears that all approaches (naive DFT, reference
// muFFT implementation, and my FFT) will return identical results if the spectrum was actually generated
// by a real-valued forward FFT.
//

FFT :: #import,file "../module.jai";

SEED_FOR_CORRECTNESS_TESTING :: 1337;
PRINT_ARRAYS :: false;
BENCHMARK_ITERS :: 100;

assert_allclose :: (x: [] $T, target: [] T, N := 1, epsilon := 1.0e-3, header := "Output mismatch:") {
  norm :: (x: Complex) -> float { return length(x); }
  norm :: (x: float32) -> float { return abs(x); }

  for 0..x.count-1 {
    diff := norm(x[it] - target[it]) / N;
    assert(diff < epsilon, "\n%\n  correct   = %\n  incorrect = %", header, target, x);
  }
}

run_test :: (N: int, $T_signal: Type, simd_support: FFT.SIMD_Support, correct_output: [] Complex) {
  print("  FFT: signal type=%, SIMD=%: ", T_signal, simd_support); 

  input,           _m1 := NewArray(N, T_signal, alignment=32);
  output,          _m2 := NewArray(N, Complex,  alignment=32);
  recovered_input, _m3 := NewArray(N, T_signal, alignment=32);
  input_copy,      _m4 := NewArray(N, T_signal, alignment=32);
  defer { free(_m1); free(_m2); free(_m3); free(_m4); }

  // Make sure we can work with transform prepared for size N/2, for real transform size N
  fft_data := FFT.prepare_fft(#ifx T_signal==float32 then N/2 else N); 
  defer FFT.free_fft(*fft_data);
  FFT.simd_support = simd_support;

  // First, check correctness:
  initialize_array_with_random_values(input, SEED_FOR_CORRECTNESS_TESTING);
  for input  input_copy[it_index] = input[it_index];
  if PRINT_ARRAYS  print("\n\ninput: %\n", input);

  FFT.forward(*fft_data, input, output);
  if PRINT_ARRAYS  print("output: %\n", output);
  assert_allclose(output, correct_output, N, header="Output spectrum mismatch:");

  FFT.inverse(*fft_data, output, recovered_input);
  if PRINT_ARRAYS  print("recovered_input: %\n\n", recovered_input);

  assert_allclose(recovered_input, input_copy, N, header="Recovered input mismatch:");
  print("PASS. ");

  // Second, benchmark:
  #if !PRINT_ARRAYS {
    fft_time := cast(float64) 0;
    for 0..BENCHMARK_ITERS-1 {
      initialize_array_with_random_values(input, xx it);
      timer_start();
      FFT.forward(*fft_data, input, output);
      fft_time += timer_stop();
    }
    print("forward [%us], ", 1.0e6*fft_time/BENCHMARK_ITERS);

    inverse_fft_time := cast(float64) 0;
    for 0..BENCHMARK_ITERS-1 {
      initialize_array_with_random_values(output, xx it);
      timer_start();
      FFT.inverse(*fft_data, output, recovered_input);
      inverse_fft_time += timer_stop();
    }
    print("inverse [%us].\n", 1.0e6*inverse_fft_time/BENCHMARK_ITERS);
  }

}

main :: () {
  using FFT;

  for log2(2)..log2(8192) {
    N := 1 << it;
    print("--------------------------------------------------\n");
    print("N = %: ", N);

    // Do a naive DFT at this size to generate the correct spectrum
    dft_real_input, _m1    := NewArray(N, float32, alignment=32);
    dft_complex_input, _m2 := NewArray(N, Complex, alignment=32);
    correct_real_output, _m3    := NewArray(N, Complex, alignment=32);
    correct_complex_output, _m4 := NewArray(N, Complex, alignment=32);
    defer { free(_m1); free(_m2); free(_m3); free(_m4); }

    initialize_array_with_random_values(dft_real_input, SEED_FOR_CORRECTNESS_TESTING);
    initialize_array_with_random_values(dft_complex_input, SEED_FOR_CORRECTNESS_TESTING);

    timer_start();
    naive_forward_real_dft(dft_real_input, correct_real_output);
    real_dft_time := timer_stop();
    print("real DFT [%us], ", 1.0e6*real_dft_time);

    timer_start();
    naive_forward_dft(dft_complex_input, correct_complex_output);
    complex_dft_time := timer_stop();
    print("complex DFT [%us]\n", 1.0e6*complex_dft_time);

    // Run FFTs in all SIMD configurations that this machine supports
    for simd_support: SIMD_Support.NONE..check_simd_support() {
      run_test(N, Complex, simd_support, correct_complex_output);
      run_test(N, float32, simd_support, correct_real_output);
    }
  }
}

initialize_array_with_random_values :: (arr: [] $T, seed: u64 = SEED_FOR_CORRECTNESS_TESTING) {
  Random.random_seed(seed);
  for 0..arr.count-1 {
    #if T == Complex {
      arr[it].x = Random.random_get_within_range(-2.5, 2.5); 
      arr[it].y = Random.random_get_within_range(-2.5, 2.5); 
    } else {
      arr[it] = Random.random_get_within_range(-2.5, 2.5);
    }
  }
}

timer_start_time: Apollo_Time;
timer_start :: () { timer_start_time = current_time_monotonic(); }
timer_stop :: () -> float64 { return to_float64_seconds(current_time_monotonic() - timer_start_time); }

log2 :: FFT.log2;
FFT_Data :: FFT.FFT_Data;

#import,file "reference_slow_ffts.jai";

#import "Basic";
#import "Math";
Random :: #import "Random";
