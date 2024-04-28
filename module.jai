#module_parameters(PRECOMPUTED_GLOBAL_FFT_SIZE := 0, DISABLE_ASSERTS := false);

// These constants are named here for clarity, but they cannot be changed: this FFT implementation
// is hardcoded for radix-8, with a base case DFT at size 4 and below.
RADIX :: 8;  
LOG2_RADIX :: #run log2(RADIX);
BASE_CASE_MAX_SIZE :: 4;

SIMD_Support :: enum { UNKNOWN; NONE; SSE; SSE3; AVX; AVX512; }
simd_support := SIMD_Support.UNKNOWN;  // Set on first forward() or inverse() call

FFT_Data :: struct { 
  forward_twiddles, inverse_twiddles, real_signal_twiddles: [] Complex;
  max_complex_transform_size: int;
}

//
// The easy, global API (just calls the normal API functions with a global FFT_Data struct.)
//
forward :: (signal: [] $T, spectrum: [] Complex) { forward(*global_fft, signal, spectrum); }
inverse :: (spectrum: [] Complex, signal: [] $T) { inverse(*global_fft, spectrum, signal); }

// Jai is awesome! We can bake precomputed FFT_Data structs into the executable.
#if PRECOMPUTED_GLOBAL_FFT_SIZE > 0 {
  global_fft :: #run prepare_fft(PRECOMPUTED_GLOBAL_FFT_SIZE, true);
} else {
  global_fft: FFT_Data;
}

//
// Normal FFT API: forward() and inverse(); prepare_fft() and free_fft().
//
forward :: (fft_data: *FFT_Data, signal: [] $T, spectrum: [] Complex) {
  run_fft(fft_data, signal, spectrum, true);
}

inverse:: (fft_data: *FFT_Data, spectrum: [] Complex, signal: [] $T) {
  run_fft(fft_data, signal, spectrum, false);
}

prepare_fft :: (size: int, called_at_compiletime := false) -> FFT_Data {
  #if !DISABLE_ASSERTS  assert(size == next_power_of_2(size), "FFT size must be a power of 2");

  using fft_data: FFT_Data;
  max_complex_transform_size = size;

  // These are the main FFT twiddles. They are stored with some redundancy, but in an order that makes
  // for efficient SIMD loads.
  forward_twiddles = precompute_twiddles(max_complex_transform_size, true);
  inverse_twiddles = precompute_twiddles(max_complex_transform_size, false);

  real_signal_twiddles = NewArray(size, Complex);
  for 0..real_signal_twiddles.count-1  real_signal_twiddles[it] = unit_polar(-TAU*it/(2*size));

  if !called_at_compiletime  simd_support = check_simd_support();

  return fft_data;
}

free_fft :: (using fft_data: *FFT_Data) { 
  array_free(forward_twiddles);
  array_free(inverse_twiddles);
  array_free(real_signal_twiddles);
}


#scope_file

//
// This function takes parameters from the above API functions, and performs the required FFT, handling
// real/complex signals, forward/inverse transforms, and various input sizes.
//
// It uses the Stockham autosort algorithm, using the input buffer as temporary workspace to calculate
// the outputs, requiring no other memory accesses (besides the precomputed twiddle factors.)
//
run_fft :: (using fft_data: *FFT_Data, signal: [] $T, spectrum: [] Complex, $forward: bool) {
  // This check will only ever happen if you use the easy global API.
  if simd_support == .UNKNOWN  simd_support = check_simd_support();

  #if !DISABLE_ASSERTS {
    assert(signal.count == next_power_of_2(signal.count), 
           "signal length % is not a power of 2!", signal.count);
    assert(signal.count == spectrum.count,
           "signal length % does not equal spectrum length %!", signal.count, spectrum.count);
    max_allowed := max_complex_transform_size << #ifx T==float32 then 1 else 0;
    assert(signal.count <= max_allowed,
           "signal length % is larger than the max allowed (%)!", signal.count, max_allowed);
  }

  #if forward {
    in  := cast(*Complex) signal.data;
    out := spectrum.data;
  } else {
    in  := spectrum.data;
    out := cast(*Complex) signal.data;
  }
  total_input_size := signal.count;

  outer_transform_size := total_input_size;
  #if T == float  outer_transform_size >>= 1;

  // Do the first butterfly if we are doing a real-valued inverse transform.
  #if !forward && T == float {
    prepare_real_inverse(fft_data, signal.data, spectrum.data, signal.count);
  }

  // Plan the FFT steps, and figure out what size the base case will be.
  base_case_size := outer_transform_size;
  kernel_steps := 0;
  while base_case_size > BASE_CASE_MAX_SIZE {
    base_case_size >>= LOG2_RADIX;
    kernel_steps += 1;
  }

  // Do the Stockham autosort algorithm, shuffling the order to prepare for the base case.
  for step: 0..kernel_steps-1 {
    N := outer_transform_size >> (step*LOG2_RADIX);
    autosort_radix8(fft_data, in, out, N, outer_transform_size);
    Swap(*in, *out);
  }

  // Execute the base-case hardcoded DFT.
  #if forward {
    base_case_forward_dft(fft_data, in, out, base_case_size, outer_transform_size);
  } else {
    base_case_inverse_dft(fft_data, in, out, base_case_size, outer_transform_size);
  }

  // Do the main FFT algorithm, the 'butterfly.'
  for < step: kernel_steps-1..0 { // Note: Reversed step order, unwinding "recursion"
    Swap(*in, *out);

    N := outer_transform_size >> (step*LOG2_RADIX);
    B := N >> LOG2_RADIX;

    // Dispatch a kernel depending on CPU featureset and current block size.
    if        simd_support == .AVX512 && B >= 8 {
      butterfly_radix8_simd(fft_data, in, out, N, outer_transform_size, forward, .AVX512);
    } else if simd_support >= .AVX    && B >= 4 {
      butterfly_radix8_simd(fft_data, in, out, N, outer_transform_size, forward, .AVX);
    } else if simd_support >= .SSE3   && B >= 2 {
      butterfly_radix8_simd(fft_data, in, out, N, outer_transform_size, forward, .SSE3_128);
    } else if simd_support >= .SSE3 {
      butterfly_radix8_simd(fft_data, in, out, N, outer_transform_size, forward, .SSE3_64);
    } else {
      // Worst case fallback, we're not running on SSE3+. (Our SSE2 impl is slower than this version.)
      butterfly_radix8_any_cpu(fft_data, in, out, N, outer_transform_size, forward);
    }
  }

  // Do the last butterfly step for a real-valued signal.
  #if forward && T == float {
    finalize_real_forward(fft_data, signal.data, spectrum.data, signal.count);
  }

  // Scale inverse transform by 1/N, as is customary.
  #if !forward {
    for 0..outer_transform_size-1  out[it] = out[it] * (1.0 / outer_transform_size);
  }
}

autosort_radix8 :: (fft_data: *FFT_Data, in: *Complex, out: *Complex, N: int, total_input_size: int) {
  n_iters := total_input_size / N;  // How many packed FFTs is this call doing?
  R :: 8;
  B := N >> LOG2_RADIX;

  for block: 0..n_iters-1 { 
    in_block  := in + N*block;
    out_block := out + N*block;

    for k: 0..B-1 {
      for r: 0..R-1 {
        out_block[k + r*B] = in_block[R*k + r];
      }
    }
  }
}

// The SIMD, inline assembly version. 
butterfly_radix8_simd :: (fft_data: *FFT_Data, in: *Complex, out: *Complex, N: int, total_input_size: int, forward := true, $simd_mode: Mode) {
  // For readability, bake the mode argument (compiletime) into each SIMD func.
  add   :: #bake_arguments complex_add(mode = simd_mode);
  sub   :: #bake_arguments complex_sub(mode = simd_mode);
  cmul  :: #bake_arguments complex_mul(mode = simd_mode);
  mov   :: #bake_arguments complex_mov(mode = simd_mode);
  load  :: #bake_arguments complex_load(mode = simd_mode);
  store :: #bake_arguments complex_store(mode = simd_mode);

  // This stuff is slightly hacky, here's the explanation:
  // For the inverse transform, we need to make 2 changes to the twiddles:
  // 1. The stored array of precomputed twiddles should be computed without the minus
  //    sign in the exponential. (Handled in precompute_twiddles().)
  // 2. The hardcoded butterfly twiddles need to have every wfwd multiplied by j,
  //    and every j multiplied by -1.
  // Luckily, this does not require actually modifying any of the assembly - it just
  // means we need to load modified values where we previously loaded wfwd and j.
  j  :: Complex.{0.0, 1.0};
  nj :: Complex.{0.0, -1.0};
  positive_j_array :: Complex.[ j,  j,  j,  j,  j,  j,  j,  j ];
  negative_j_array :: Complex.[ nj, nj, nj, nj, nj, nj, nj, nj ];
  wfwd :: #run unit_polar(-TAU*1/8);
  winv :: #run j*unit_polar(-TAU*1/8);
  wfwd_array :: Complex.[ wfwd, wfwd, wfwd, wfwd, wfwd, wfwd, wfwd, wfwd ];
  winv_array :: Complex.[ winv, winv, winv, winv, winv, winv, winv, winv ];

  j_array    := ifx forward then positive_j_array else negative_j_array;
  w1_8_array := ifx forward then wfwd_array else winv_array;

  N_div_2 := N >> 1;
  n_iters := total_input_size / N;  // How many FFTs is this call doing?

  #if simd_mode == .AVX512                                  log2_simd_width := 3;
  else #if simd_mode == .AVX                                log2_simd_width := 2;
  else #if simd_mode == .SSE3_128 || simd_mode == .SSE_128  log2_simd_width := 1;
  else                                                      log2_simd_width := 0;

  R :: 8;
  B := N >> LOG2_RADIX;
  B_div_simd_width := B >> log2_simd_width;
  log2_B := log2(B);

  twiddles := ifx forward fft_data.forward_twiddles.data else fft_data.inverse_twiddles.data;
  twiddles += twiddle_offset_for_size(N);

  for iter: 0..n_iters-1 {   // Outer loop: do several consecutive butterflies
    for b_div_simd_width: 0..B_div_simd_width-1 { 
      b := b_div_simd_width << log2_simd_width;
      iarr := N*iter + b;

      // 4 registers for outputs, 8..? for temp values
      #asm {
        out0: vec; out1: vec; out2: vec; out3: vec; 
        t0: vec; t1: vec; t2: vec; t3: vec; t4: vec; t5: vec; t6: vec; t7: vec;
        constant: vec;  // Used for imaginary unit and root-2 twiddle at different times
      }

      // We'll need the imaginary constant a few times before we need w1_8
      load(constant, j_array.data);

      // Load all twiddles into temp registers
      //load(t0, twiddles + 0*B + b);  // This is all ones
      load(t1, twiddles +   B + b);
      load(t2, twiddles + 2*B + b);
      load(t3, twiddles + 3*B + b);
      load(t4, twiddles + 4*B + b);
      load(t5, twiddles + 5*B + b);
      load(t6, twiddles + 6*B + b);
      load(t7, twiddles + 7*B + b);

      // Load first half of inputs, do twiddle multiplications where necessary
      load(out0, in + iarr);
      load(out1, in + iarr +   B);
      load(out2, in + iarr + 2*B);
      load(out3, in + iarr + 3*B);

      mov(t0, out0); // Skip multiplying by 1
      //cmul(t0, t0, out0);
      cmul(t1, t1, out1);
      cmul(t2, t2, out2);
      cmul(t3, t3, out3);

      // Load second half of inputs, do twiddle multiplications
      load(out0, in + iarr + 4*B);
      load(out1, in + iarr + 5*B);
      load(out2, in + iarr + 6*B);
      load(out3, in + iarr + 7*B);

      cmul(t4, t4, out0);
      cmul(t5, t5, out1);
      cmul(t6, t6, out2);
      cmul(t7, t7, out3);

      // Now combine our temporary values into add/sub pairs. 
      // (The out registers are now free for temporary use themselves.)
      mov(out0, t0);
      mov(out1, t4);
      mov(out2, t1);
      mov(out3, t5);
      add(t0, out0, out1);  // t0 is now t0p4
      sub(t4, out0, out1);  // t4 is now t0m4
      add(t1, out2, out3);  // etc....
      sub(t5, out2, out3);

      mov(out0, t2);
      mov(out1, t6);
      mov(out2, t3);
      mov(out3, t7);
      add(t2, out0, out1);
      sub(t6, out0, out1);
      add(t3, out2, out3); 
      sub(t7, out2, out3);

      // Now combine outputs: for each output pair, use out0 and out1 as the actual output
      // registers, and use out2 and out3 as fst and snd
      
      // Outputs 0 and 4
      add(out2, t0, t2);
      add(out3, t1, t3);
      add(out0, out2, out3);
      sub(out1, out2, out3);
      store(out + iarr,       out0);
      store(out + iarr + 4*B, out1);

      // Outputs 2 and 6
      sub(out2, t0, t2);
      sub(out3, t3, t1);
      cmul(out3, out3, constant);
      add(out0, out2, out3);
      sub(out1, out2, out3);
      store(out + iarr + 2*B, out0);
      store(out + iarr + 6*B, out1);

      // This will be used only with the j-mult 
      cmul(t6, t6, constant);

      // Outputs 1 and 5
      sub(out2, t4, t6);
      load(constant, j_array.data);
      cmul(t0, constant, t7); // temp t0 for j*t3m7
      sub(t1, t5, t0); // temp t1 for t1m5 - j*t3m7
      load(constant, w1_8_array.data);
      cmul(out3, t1, constant); 
      add(out0, out2, out3);
      sub(out1, out2, out3);
      store(out + iarr +   B, out0);
      store(out + iarr + 5*B, out1);

      // Outputs 3 and 7
      add(out2, t4, t6);
      load(constant, j_array.data);
      cmul(t0, t5, constant); // tmp t0 for j*t1m5
      sub(t1, t7, t0); /// tmp t1 for t3m7 - j*t1m5
      load(constant, w1_8_array.data);
      cmul(out3, constant, t1);
      add(out0, out2, out3);
      sub(out1, out2, out3);
      store(out + iarr + 3*B, out0);
      store(out + iarr + 7*B, out1);
    }
  }
}

//
// This is the scalar version of the above function. It gets called if we aren't running on x86_64, or
// also actually if we only support SSE, because it's hard to optimize complex multiplication without SSE3.
//
butterfly_radix8_any_cpu:: (fft_data: *FFT_Data, in: *Complex, out: *Complex, N: int, total_input_size: int, forward := true) {
  n_iters := total_input_size / N;  // How many FFTs is this call doing?

  R :: 8;
  B := N >> LOG2_RADIX;
  log2_B := log2(B);

  radix_mask := B - 1;

  twiddles := ifx forward fft_data.forward_twiddles.data else fft_data.inverse_twiddles.data;
  twiddles += twiddle_offset_for_size(N);

  total_iters := total_input_size >> LOG2_RADIX;

  for outer_iter: 0..total_iters-1 {
    iter := outer_iter >> log2_B;
    b := outer_iter & radix_mask;

    N_iter_b := N*iter + b; 
    b_mod := b & radix_mask;

    iarr := N*iter + b;

    in0 := in[iarr + 0*B];
    in1 := in[iarr + 1*B];
    in2 := in[iarr + 2*B];
    in3 := in[iarr + 3*B];
    in4 := in[iarr + 4*B];
    in5 := in[iarr + 5*B];
    in6 := in[iarr + 6*B];
    in7 := in[iarr + 7*B];

    w1 := twiddles[B*1 + b];
    w2 := twiddles[B*2 + b];
    w3 := twiddles[B*3 + b];
    w4 := twiddles[B*4 + b];
    w5 := twiddles[B*5 + b];
    w6 := twiddles[B*6 + b];
    w7 := twiddles[B*7 + b];

    t0 := in0;
    t1 := w1*in1;
    t2 := w2*in2;
    t3 := w3*in3;
    t4 := w4*in4;
    t5 := w5*in5;
    t6 := w6*in6;
    t7 := w7*in7;

    // These constants multiply certain twiddles for the final output
    //j :: #run imag(1);
    pj :: Complex.{0.0, 1.0};
    nj :: Complex.{0.0, -1.0};

    wfwd :: #run unit_polar(-TAU*1/8);    // sqrt(2) - sqrt(2)*j
    winv :: #run unit_polar(TAU*1/8);  
    w1_8 := ifx forward then wfwd else winv;
    j    := ifx forward then pj else nj;

    t0p4 := in0 + t4;
    t0m4 := in0 - t4;
    t1p5 := t1 + t5;
    t1m5 := t1 - t5;
    t2p6 := t2 + t6;
    t2m6 := t2 - t6;
    t3p7 := t3 + t7;
    t3m7 := t3 - t7;

    {
      fst := t0p4 + t2p6;
      snd := t1p5 + t3p7;
      out[iarr + 0*B] = fst + snd;
      out[iarr + 4*B] = fst - snd;
    }

    {
      fst := t0p4 - t2p6;
      snd := j*(-t1p5 + t3p7);
      out[iarr + 2*B] = fst + snd;
      out[iarr + 6*B] = fst - snd;
    }

    {
      fst := t0m4 - j*t2m6;
      snd := w1_8*(t1m5 - j*t3m7);
      out[iarr + 1*B] = fst + snd;
      out[iarr + 5*B] = fst - snd;
    }

    {
      fst := t0m4 + j*t2m6;
      snd := w1_8*(-j*t1m5 + t3m7);
      out[iarr + 3*B] = fst + snd;
      out[iarr + 7*B] = fst - snd;
    }
  }
}

//
// These base case functions just perform the simple, naive DFT at size 1, 2, or 4.
// All the twiddle factors are +/- 1 or j, so we don't have to do actual complex multiplication.
//
base_case_forward_dft :: (fft_data: *FFT_Data, in: *Complex, out: *Complex, N: int, total_input_size: int) {
  n_iters := total_input_size / N;

  if N == 1 {
    memcpy(out, in, total_input_size*size_of(Complex));
  }
  else if N == 2 { 
    for 0..n_iters-1 {
      fst := in[2*it];
      snd := in[2*it + 1];
      out[2*it]     = fst + snd;
      out[2*it + 1] = fst - snd;
    }
  }
  else if N == 4 { 
    for 0..n_iters-1 {
      x := in  + N*it;
      X := out + N*it;
      X[0] = x[0] + x[1] + x[2] + x[3];
      X[1] = x[0] + .{x[1].y, -x[1].x} - x[2] + .{-x[3].y, x[3].x};
      X[2] = x[0] - x[1] + x[2] - x[3];
      X[3] = x[0] + .{-x[1].y, x[1].x} - x[2] + .{x[3].y, -x[3].x};
    }
  }
}

base_case_inverse_dft :: (fft_data: *FFT_Data, in: *Complex, out: *Complex, N: int, total_input_size: int) {
  n_iters := total_input_size / N;

  if N == 1 {
    memcpy(out, in, total_input_size*size_of(Complex));
  }
  else if N == 2 { 
    for 0..n_iters-1 {
      fst := in[2*it];
      snd := in[2*it + 1];
      out[2*it]     = fst + snd;
      out[2*it + 1] = fst - snd;
    }
  }
  else if N == 4 {  
    for 0..n_iters-1 {
      x := in  + N*it;
      X := out + N*it;
      X[0] = x[0] + x[1] + x[2] + x[3];
      X[1] = x[0] + .{-x[1].y, x[1].x} - x[2] + .{x[3].y, -x[3].x};
      X[2] = x[0] - x[1] + x[2] - x[3];
      X[3] = x[0] + .{x[1].y, -x[1].x} - x[2] + .{-x[3].y, x[3].x};
    }
  }
}

// 1. Cast the real buffer to complex.
// 2. Do a N/2 forward FFT on the buffer.
// 3. Call this function to shuffle the outputs.
//
// Note: This function signature looks a little weird when you call it. It looks like it
// uses 'signal' as its input, but it does not - it only reads from 'spectrum', and uses
// 'signal' as temporary workspace before writing the output back to 'spectrum'.
//
finalize_real_forward :: (using fft: *FFT_Data, signal: *float32, spectrum: *Complex, N_real: int) {
  // Per blogpost in README.md:
  // - do FFT, interpreting size-N real 'signal' as size-N/2 complex
  // - do the following postprocessing:
  //   Xe[k] = 0.5  * (Z[k] + conj(Z[N/2 - k]))
  //   Xo[k] = 0.5j * (Z[k] - conj(Z[N/2 - k]))
  //   for k: 0..N/2-1    X[k]  = Xe[k] + Xo[k] * fwd_twiddle(k, N/2)
  //   for k = N/2        X[k]  = Xe[k] + Xo[k]
  //   for k: N/2+1..N-1  X[k]  = conj(X[N - k])

  // In order to avoid extra storage or allocations, we use the signal buffer as extra workspace, 
  // much like the Stockham autosort algorithm in the main FFT function.

  // The one extra consideration here is, we are reading the array forward and backward at the same
  // time in order to write forward. So, in order to avoid overwriting data that we will need later,
  // we actually do entries at a time, reading and writing one forward and one backward per iteration.

  N := N_real;
  N_div_2 := N >> 1;

  Xe := cast(*Complex) signal;
  Xo := spectrum;

  // Tricky: this needs to run up to index N/4, not N/4-1.
  for k: 0..(N_div_2 >> 1) {
    Z_ascending  := spectrum[k];
    Z_descending := spectrum[(N_div_2 - k) & (N_div_2 - 1)];

    {
      fst := Z_ascending;
      snd := conj(Z_descending);
      Xe[k] =  0.5   * (fst + snd);
      Xo[k] = -0.5*j * (fst - snd);
    }

    {
      fst := Z_descending;
      snd := conj(Z_ascending);
      Xe[(N_div_2 - k) & (N_div_2 - 1)] =  0.5   * (fst + snd);
      Xo[(N_div_2 - k) & (N_div_2 - 1)] = -0.5*j * (fst - snd);
    }
  }

  stride := 2*max_complex_transform_size/N;

  for k: 0..N_div_2-1 {
    fst := Xe[k];
    snd := Xo[k]*real_signal_twiddles[k*stride];

    spectrum[k] = fst + snd;
    if k == 0  spectrum[N_div_2] = fst - snd;  // Only need this one conjugate at Nyquist
  }
}

prepare_real_inverse :: (using fft: *FFT_Data, signal: *float32, spectrum: *Complex, N_real: int) {
  N := N_real;
  N_div_2 := N >> 1;

  Xe := cast(*Complex) signal;   // Future inplace Stockham ish
  Xo := spectrum;

  stride := 2*max_complex_transform_size/N;

  // Similar to finalize_real_inverse, this has to write in both directions at once.
  for k: 0..(N_div_2 >> 1) {  
    Z_ascending  := spectrum[k];
    Z_descending := spectrum[N_div_2 - k];

    {
      fst := Z_ascending;
      snd := conj(Z_descending);
      Xe[k] = 0.5*(fst + snd);
      Xo[k] = 0.5*(fst - snd)*conj(real_signal_twiddles[stride*k]);
    }

    {
      fst := Z_descending;
      snd := conj(Z_ascending);
      
      w2 := conj(real_signal_twiddles[stride * ((N_div_2 - k) & (N_div_2 - 1))]);
      if k == 0  w2 = -w2;

      Xe[(N_div_2 - k) & (N_div_2 - 1)] = 0.5*(fst + snd);
      Xo[(N_div_2 - k) & (N_div_2 - 1)] = 0.5*(fst - snd)*w2;
    }
  }

  for k: 0..N/2-1  spectrum[k] = Xe[k] + j*Xo[k];
}

precompute_twiddles:: (N: int, forward: bool) -> [] Complex {
  twiddles := NewArray(2*N, Complex);

  log_N := log2(N);
  for log_this_N: 0..log_N {
    this_N := 1 << log_this_N;
    //print("Precomputing twiddles for N = %\n", this_N); 

    B := this_N >> 3;

    offset := twiddle_offset_for_size(this_N);
    for r: 0..7 {
      for b: 0..B-1 {
        sign := ifx forward then -1 else 1;
        index := B*r + b;
        twiddles[offset + index] = unit_polar(sign*TAU*(r*b)/this_N);
        //print("  twiddles[%] (overall index %) = %\n", index, r*b, twiddles[offset+index]);
      }
    }
  }

  return twiddles;
}


#scope_export

// These little helpers may be useful to higher level code using the FFT, like convolutions,
// spectrum analyzers, etc.

check_simd_support :: () -> SIMD_Support {
  cpu_features := x64.get_cpu_info().feature_leaves;
  if x64.check_feature(cpu_features, x64.x86_Feature_Flag.AVX512F) return .AVX512;
  if x64.check_feature(cpu_features, x64.x86_Feature_Flag.AVX)     return .AVX;
  if x64.check_feature(cpu_features, x64.x86_Feature_Flag.SSE3)    return .SSE3;
  if x64.check_feature(cpu_features, x64.x86_Feature_Flag.SSE)     return .SSE;
  return .NONE;
}

log2 :: (x: $T) -> T {  y := 0;  while 1 << y < x  y += 1;  return y;  }
next_power_of_2 :: (x: $T) -> T {  y := 1;  while y < x  y <<= 1;  return y;  }
unit_polar :: (theta: float) -> Complex { return .{ cos(theta), sin(theta) }; }
conj :: (z: Complex) -> Complex #expand { return .{ z.x, -z.y }; }
j :: Complex.{0.0, 1.0};


#scope_file 

#import "Basic";
#import "Math";
x64 :: #import "Machine_X64";

#load "simd_helpers.jai";

// This is obviously dumb, but if we were really going to optimize storage, the 
// offset wouldn't just be N. Maybe that will happen in the future.
twiddle_offset_for_size :: (N: int) -> int #expand  { return N; }

