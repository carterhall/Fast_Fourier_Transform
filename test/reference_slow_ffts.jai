//
// This file is full of a bunch of different transform implementations that I wrote on the 
// way to the final versions.
//
// An actual optimized FFT is pretty hard to read, so it seemed like a good idea to leave
// this progression of FFTs in the repo somewhere.
//

// These naive DFTs were validated against numpy.fft's fft() and ifft(), with
// N=8. See reference.py.
naive_forward_dft :: (x: [] Complex, X: [] Complex) {
  for k: 0..x.count-1 {
    X[k] = .{0.0, 0.0};
    for n: 0..x.count-1 {
      twiddle := unit_polar(-TAU*k*n/x.count);
      //print("fwd DFT twiddle[%][%]: %\n", k, n, twiddle);
      X[k] += twiddle * x[n];
    }
  }
}

naive_inverse_dft :: (X: [] Complex, x: [] Complex) {
  for n: 0..x.count-1 {
    x[n] = .{0.0, 0.0};
    for k: 0..x.count-1 {
      twiddle := unit_polar(TAU*k*n/x.count);
      //print("inv DFT twiddle[%][%]: %\n", n, k, twiddle);
      x[n] += twiddle * X[k];
    }
    x[n] *= (1.0 / x.count);
  }
}

// Real-signal versions of the naive DFTs, ultimately just calling the complex versions.
// There is a little bit of subtlety in how we handle the symmetric redundant parts of
// the spectrum, to match the behavior of existing FFT libraries.

naive_forward_real_dft :: (in: [] float, out: [] Complex) {
  assert(in.count == out.count);
  N := in.count;

  in_complex: [..] Complex;
  array_resize(*in_complex, N);
  defer array_free(in_complex);
  for 0..in.count-1  in_complex[it] = .{ in[it], 0.0 };
  naive_forward_dft(in_complex, out);

  // Self-consistency check: the FFT of a real-valued signal should have this symmetry,
  // where index k should be equal to the conjugate of index N - k.
  eps := 1.0e-3;
  for 1..out.count/2-1 {
    diff := out[N - it] - conj(out[it]);
    assert(abs(diff.x)/N < eps);
    assert(abs(diff.y)/N < eps);
  }

  // Now, after checking symmetry, we wipe out the redundant values, to match the
  // behavior of existing FFT libraries.
  for N/2+1..N-1  out[it] = .{ 0.0, 0.0 };
}

naive_inverse_real_dft :: (in: [] Complex, out: [] float) {
  assert(in.count == out.count);
  N := in.count;

  // If we didn't wipe out the redundant values in naive_forward_real_dft, we could
  // just run the complex inverse DFT, ignore the (~zero) imaginary parts, and be done.
  // However, if we run the iDFT on the spectrum missing those redundant values, we 
  // will actually get incorrect outputs with nonnegligible imaginary parts.

  // So, we need to preprocess the input array, recovering the redundant values.
  in_full := NewArray(N, Complex);
  defer array_free(in_full);

  in_full[0] = in[0];
  in_full[N/2] = in[N/2];

  for 1..N/2-1 {
    in_full[it]     = in[it];
    in_full[N - it] = conj(in[it]);
  }

  out_complex := NewArray(N, Complex);
  defer array_free(out_complex);

  naive_inverse_dft(in_full, out_complex);
  for 0..N-1  out[it] = out_complex[it].x;
}

// Faster base case DFTs for the small sizes
forward_base_case :: (x: *Complex, X: *Complex, N: int) {
  assert(N == 1 || N == 2 || N == 4);

  if N == 1 {
    X[0] = x[0];
  }
  else if N == 2 {
    X[0] = x[0] + x[1];
    X[1] = x[0] - x[1];
  }
  else if N == 4 {
    X[0] = x[0] + x[1] + x[2] + x[3];
    X[1] = x[0] + .{x[1].y, -x[1].x} - x[2] + .{-x[3].y, x[3].x};
    X[2] = x[0] - x[1] + x[2] - x[3];
    X[3] = x[0] + .{-x[1].y, x[1].x} - x[2] + .{x[3].y, -x[3].x};
  }
}

inverse_base_case :: (X: *Complex, x: *Complex, N: int) {
  assert(N == 1 || N == 2 || N == 4);

  if N == 1 {
    x[0] = X[0];
  }
  else if N == 2 {
    x[0] = X[0] + X[1];
    x[1] = X[0] - X[1];
  }
  else if N == 4 {
    x[0] = X[0] + X[1] + X[2] + X[3];
    x[1] = X[0] + .{-X[1].y, X[1].x} - X[2] + .{X[3].y, -X[3].x};
    x[2] = X[0] - X[1] + X[2] - X[3];
    x[3] = X[0] + .{X[1].y, -X[1].x} - X[2] + .{-X[3].y, X[3].x};
  }
}

//
// The simplest FFT: Radix-2 decimation in time.
//
// The only optimization is that it uses the input buffer for intermediate
// calculations, so no allocations are required.
//
// Doesn't use the FFT_Setup argument; it is just there to match the API.
//
forward_radix2 :: (in: [] Complex, out: [] Complex) {
  if in.count <= 4 {
    forward_base_case(in.data, out.data, in.count);
    return;
  }

  N      := in.count;
  half_N := N / 2;

  out_evens := get_sub_array(in, 0, half_N);
  out_odds  := get_sub_array(in, half_N, half_N);
  in_evens := get_sub_array(out, 0, half_N);
  in_odds  := get_sub_array(out, half_N, half_N);

  for k: 0..half_N-1  in_evens[k] = in[2*k];
  for k: 0..half_N-1  in_odds[k]  = in[2*k + 1];

  forward_radix2(in_evens, out_evens);
  forward_radix2(in_odds,  out_odds);

  for k: 0..half_N-1 {
    w := unit_polar(-TAU*k/N); 
    out[k]          = out_evens[k] + w*out_odds[k];
    out[k + half_N] = out_evens[k] - w*out_odds[k];
  }
}

//
// This version adds supports for radices other than 2, but is still slow.
// It doesn't take advantage of twiddle symmetries or trivial twiddles, so it 
// ends up doing more complex multiplications than are actually necessary.
//
forward_arbitrary_radix :: (in: [] Complex, out: [] Complex) {
  radix :: 8;

  if in.count < radix {
    forward_base_case(in.data, out.data, in.count);
    return;
  }

  N  := in.count;
  N1 := radix;
  N2 := N / N1;

  // Stockham autosort: shuffle inputs into the output buffer
  for n1: 0..N1-1 {
    for n2: 0..N2-1 {
      //print("Writing to %, reading from %\n", n2 + n1*N2, N1*n2 + n1);
      out[N2*n1 + n2] = in[N1*n2 + n1];
    }
  }

  // Recursively transform from the sorted output buffer to the input buffer
  for k1: 0..N1-1 {
    sub_fft_input  := get_sub_array(out, k1*N2, N2);
    sub_fft_output := get_sub_array(in,  k1*N2, N2);
    forward_arbitrary_radix(sub_fft_input, sub_fft_output);
  }

  // Arbitrary-radix butterfly
  for k1: 0..N1-1 {
    for k2: 0..N2-1 {
      k := N2*k1 + k2;   // Linear output index k for clarity
      
      out[k] = .{0.0, 0.0};
      for n1: 0..N1-1 {
        sub_fft_output_n1 := get_sub_array(in, n1*N2, N2);
        twiddle := unit_polar(-TAU/N*n1*k); 
        out[k] = out[k] + twiddle*sub_fft_output_n1[k2];
      }
    }
  }
}

inverse_arbitrary_radix :: (in: [] Complex, out: [] Complex, scale_output := true) {
  radix :: 4;
  N  := in.count;
  N1 := radix;
  N2 := N / N1;

  if in.count < radix {
    inverse_base_case(in.data, out.data, in.count);

  } else {
    // Stockham autosort: shuffle inputs into the output buffer
    for n1: 0..N1-1 {
      for n2: 0..N2-1 {
        //print("Writing to %, reading from %\n", n2 + n1*N2, N1*n2 + n1);
        out[N2*n1 + n2] = in[N1*n2 + n1];
      }
    }

    // Recursively transform from the sorted output buffer to the input buffer
    for k1: 0..N1-1 {
      sub_fft_input  := get_sub_array(out, k1*N2, N2);
      sub_fft_output := get_sub_array(in,  k1*N2, N2);
      inverse_arbitrary_radix(sub_fft_input, sub_fft_output, scale_output=false);
    }

    // Arbitrary-radix butterfly
    for k1: 0..N1-1 {
      for k2: 0..N2-1 {
        k := N2*k1 + k2;   // Linear output index k for clarity
        
        out[k] = .{0.0, 0.0};
        for n1: 0..N1-1 {
          sub_fft_output_n1 := get_sub_array(in, n1*N2, N2);
          twiddle := unit_polar(TAU/N*n1*k); 
          out[k] = out[k] + twiddle*sub_fft_output_n1[k2];
        }
      }
    }
  }

  if scale_output {
    for 0..N-1  out[it] = out[it] * (1.0 / N);
  }
}


// A hardcoded radix-4 butterfly, much faster than the arbitrary radix one.
forward_radix4 :: (in: [] Complex, out: [] Complex) {
  radix :: 4;

  if in.count <= 4 {
    forward_base_case(in.data, out.data, in.count);
    return;
  }

  N  := in.count;
  R := radix;
  B := N / R;

  // Stockham autosort: shuffle inputs into the output buffer
  for r: 0..R-1 {
    for b: 0..B-1 {
      out[B*r + b] = in[R*b + r];
    }
  }

  // Recursively compute smaller FFTs
  for r: 0..R-1 {
    sub_fft_in  := get_sub_array(out, r*B, B);
    sub_fft_out := get_sub_array(in, r*B, B);
    forward_radix4(sub_fft_in, sub_fft_out);
  }

  // Radix-4 butterfly
  for b: 0..B-1 {
    w1 := unit_polar(-TAU*b/N);
    w2 := unit_polar(-TAU*2*b/N);
    w3 := unit_polar(-TAU*3*b/N);
    
    i0 := in[b + 0*B];
    i1 := in[b + 1*B];
    i2 := in[b + 2*B];
    i3 := in[b + 3*B];

    //j :: #run imag(1);  // Efficient-multiplier imaginary constant
    j :: Complex.{0.0, 1.0};

    out[b + 0*B] = i0 +   w1*i1 + w2*i2 +   w3*i3;
    out[b + 1*B] = i0 - j*w1*i1 - w2*i2 + j*w3*i3;
    out[b + 2*B] = i0 -   w1*i1 + w2*i2 -   w3*i3;
    out[b + 3*B] = i0 + j*w1*i1 - w2*i2 - j*w3*i3;
  }
}

//
// A hardcoded radix-8 butterfly, much faster than the arbitrary radix one.
//
// This is the basis for the full-power FFT, which does the same thing as
// this function, but 1) precomputes twiddles, 2) iterates through known steps
// instead of actually recursing, and 3) has inline assembly to vectorize
// the twiddle multiplications manually.
//
forward_radix8 :: (in: [] Complex, out: [] Complex) {
  radix :: 8;

  if in.count <= 4 {
    forward_base_case(in.data, out.data, in.count);
    return;
  }

  N  := in.count;
  R := radix;
  B := N / R;

  // Stockham autosort: shuffle inputs into the output buffer
  for r: 0..R-1 {
    for b: 0..B-1 {
      out[B*r + b] = in[R*b + r];
    }
  }

  // Recursively compute smaller FFTs
  for r: 0..R-1 {
    sub_fft_in  := get_sub_array(out, r*B, B);
    sub_fft_out := get_sub_array(in, r*B, B);
    forward_radix8(sub_fft_in, sub_fft_out);
  }

  // Radix-8 butterfly
  for b: 0..B-1 {
    w, i, t: [8] Complex;

    for 0..7  w[it] = unit_polar(-TAU*it*b/N);   // Create twiddles
    for 0..7  i[it] = in[b + it*B];              // Gather inputs 
    for 1..7  t[it] = w[it] * i[it];             // Multiply twiddles

    // These constants multiply certain twiddles for the final output
    //j :: #run imag(1);
    j :: Complex.{0.0, 1.0};
    w1_8 :: #run unit_polar(-TAU*1/8);    // sqrt(2) - sqrt(2)*j

    out[b + 0*B] = i[0] +        t[1] +   t[2] +        t[3] + t[4] +        t[5] +   t[6] +        t[7];
    out[b + 1*B] = i[0] +   w1_8*t[1] - j*t[2] - j*w1_8*t[3] - t[4] -   w1_8*t[5] + j*t[6] + j*w1_8*t[7];
    out[b + 2*B] = i[0] -      j*t[1] -   t[2] +      j*t[3] + t[4] -      j*t[5] -   t[6] +      j*t[7];
    out[b + 3*B] = i[0] - j*w1_8*t[1] + j*t[2] +   w1_8*t[3] - t[4] + j*w1_8*t[5] - j*t[6] -   w1_8*t[7];
    out[b + 4*B] = i[0] -        t[1] +   t[2] -        t[3] + t[4] -        t[5] +   t[6] -        t[7];
    out[b + 5*B] = i[0] -   w1_8*t[1] - j*t[2] + j*w1_8*t[3] - t[4] +   w1_8*t[5] + j*t[6] - j*w1_8*t[7];
    out[b + 6*B] = i[0] +      j*t[1] -   t[2] -      j*t[3] + t[4] +      j*t[5] -   t[6] -      j*t[7];
    out[b + 7*B] = i[0] + j*w1_8*t[1] + j*t[2] -   w1_8*t[3] - t[4] - j*w1_8*t[5] - j*t[6] +   w1_8*t[7];

  }
}

inverse_radix8 :: (in: [] Complex, out: [] Complex, scale_output := true) {
  radix :: 8;
  N  := in.count;
  R := radix;
  B := N / R;

  if in.count <= 4 {
    inverse_base_case(in.data, out.data, in.count);

  } else {
    // Stockham autosort: shuffle inputs into the output buffer
    for r: 0..R-1 {
      for b: 0..B-1 {
        out[B*r + b] = in[R*b + r];
      }
    }

    // Recursively compute smaller FFTs
    for r: 0..R-1 {
      sub_fft_in  := get_sub_array(out, r*B, B);
      sub_fft_out := get_sub_array(in, r*B, B);
      inverse_radix8(sub_fft_in, sub_fft_out, scale_output=false);
    }

    conj :: (c: Complex) -> Complex {
      return .{c.x, -c.y};
    }

    // Radix-8 butterfly
    for b: 0..B-1 {
      w, i, t: [8] Complex;

      for 0..7  w[it] = unit_polar(TAU*it*b/N);   // Create twiddles
      for 0..7  i[it] = in[b + it*B];              // Gather inputs 
      for 1..7  t[it] = w[it] * i[it];             // Multiply twiddles

      // These constants multiply certain twiddles for the final output
      actual_j :: Complex.{0.0, 1.0};
      j :: Complex.{0.0, -1.0};
      w1_8 :: #run actual_j * unit_polar(-TAU*1/8);    

      out[b + 0*B] = i[0] +        t[1] +   t[2] +        t[3] + t[4] +        t[5] +   t[6] +        t[7];
      out[b + 1*B] = i[0] +   w1_8*t[1] - j*t[2] - j*w1_8*t[3] - t[4] -   w1_8*t[5] + j*t[6] + j*w1_8*t[7];
      out[b + 2*B] = i[0] -      j*t[1] -   t[2] +      j*t[3] + t[4] -      j*t[5] -   t[6] +      j*t[7];
      out[b + 3*B] = i[0] - j*w1_8*t[1] + j*t[2] +   w1_8*t[3] - t[4] + j*w1_8*t[5] - j*t[6] -   w1_8*t[7];
      out[b + 4*B] = i[0] -        t[1] +   t[2] -        t[3] + t[4] -        t[5] +   t[6] -        t[7];
      out[b + 5*B] = i[0] -   w1_8*t[1] - j*t[2] + j*w1_8*t[3] - t[4] +   w1_8*t[5] + j*t[6] - j*w1_8*t[7];
      out[b + 6*B] = i[0] +      j*t[1] -   t[2] -      j*t[3] + t[4] +      j*t[5] -   t[6] -      j*t[7];
      out[b + 7*B] = i[0] + j*w1_8*t[1] + j*t[2] -   w1_8*t[3] - t[4] - j*w1_8*t[5] - j*t[6] +   w1_8*t[7];
    }
  }

  if scale_output  for 0..N-1  out[it] = out[it] * (1.0 / N);
}

// This is an easy way to calculate the twiddles that you should hardcode when 
// writing a butterfly. For R=4 they are all trivial; for R=8 they are half trivial
// and half trivial multiples of sqrt(2) - j*sqrt(2). And of course for R=2 they're 
// either 1 or -1. It starts to get ugly at R=16, but still potentially worth it.
//
// Note that these are not all the twiddles you need - these are the ones that
// multiply the stored ones to reduce the memory requirement.
print_hardcoded_twiddles_for_radix :: (R: int, forward := true) {
  print_one_twiddle :: (w: Complex, forward: bool) {
    eps := 1.0e-3;

    // First special cases: trivial twiddles 1, -1, j, -j  (from unit_polar(-TAU*r/4))
    if      abs(w.x - 1) < eps && abs(w.y) < eps  print("1");
    else if abs(w.x + 1) < eps && abs(w.y) < eps  print("-1");
    else if abs(w.y - 1) < eps && abs(w.x) < eps  print("j");
    else if abs(w.y + 1) < eps && abs(w.x) < eps  print("-j");

    // Second round of special cases, from unit_polar(-TAU*r/8)
    else if abs(w.x) - sqrt(2.0) < eps && abs(w.y) - sqrt(2.0) < eps {
      //w1_8 := ifx forward then unit_polar(-TAU*1/8) else unit_polar(TAU*1/8);
      w1_8 := unit_polar(-TAU*1/8);
      print_one_twiddle(w / w1_8, forward);
      print("*w1_8");
    }

    // Otherwise just print the raw complex number
    else {
      print("%", w);
    }
  }

  multiplier := ifx forward then -TAU else TAU;
  for r_out: 0..R-1 {
    print("r_out=%: ", r_out);
    for r_in: 0..R-1 {
      print_one_twiddle(unit_polar(multiplier*r_in*r_out/R), forward);
      print(", ");
    }
    print("\n");
  }
}


/*
// These don't really belong anywhere, and aren't useful by themselves, and were 
// an intermediate step to the radix8 version...
autosort_radix2 :: (fft_data: *FFT_Data($T), in: *Complex, out: *Complex, step: int) {
  N := fft_data.N >> (step * fft_data.SHIFT_AMOUNT);
  half_N := N >> 1;   
  n_iters := fft_data.N / N;  // How many FFTs is this call doing?

  for block: 0..n_iters-1 { 
    in_block  := in + N*block;
    out_block := out + N*block;

    for k: 0..half_N-1 {
      out_block[k]          = in_block[2*k];
      out_block[k + half_N] = in_block[2*k + 1];
    }
  }
}

butterfly_radix2 :: (fft_data: *FFT_Data($T), in: *Complex, out: *Complex, step: int, forward := true) {
  N := fft_data.N >> (step * fft_data.SHIFT_AMOUNT);
  half_N := N >> 1; 
  n_iters := fft_data.N / N;  // How many FFTs is this call doing?

  twiddle_stride := 1 << (step * fft_data.SHIFT_AMOUNT);
  twiddles := ifx forward fft_data.twiddles_forward else fft_data.twiddles_inverse;

  for block: 0..n_iters-1 { 
    in_block  := in + N*block;
    out_block := out + N*block;

    for k: 0..half_N-1 {
      twiddle := twiddles[k*twiddle_stride];
      //twiddle := unit_polar(-TAU*k/N);
      out_block[k]          = in_block[k] + twiddle*in_block[k + half_N];
      out_block[k + half_N] = in_block[k] - twiddle*in_block[k + half_N];
    }
  }

  if fft_data.forward_scalar != 1.0 {
    for 0..fft_data.N-1  out[it] = out[it]*fft_data.forward_scalar;
  }
}
*/



butterfly_radix8_basic :: (fft_data: *FFT_Data, in: *Complex, out: *Complex, step: int, forward := true) {
  N := fft_data.max_complex_transform_size >> (step * LOG2_RADIX);
  n_iters := fft_data.max_complex_transform_size / N;  // How many FFTs is this call doing?

  index_mask := fft_data.max_complex_transform_size - 1;  // Cheap power-of-2 modulo

  R :: 8;
  B := N >> LOG2_RADIX;

  twiddles := ifx forward fft_data.forward_twiddles.data else fft_data.inverse_twiddles.data;
  twiddles += twiddle_offset_for_size(N);

  for iter: 0..n_iters-1 {   // Outer loop: do several consecutive butterflies
    for b: 0..B-1 {
      w, i, t: [8] Complex;
      iarr := N*iter + b;
      N_div_2 := N >> 1;

      /*
      print("basic butterfly: B=8: array accesses: ");
      for 0..7 print("%, ", iarr + it*B);
      print("\n");
      */
      /*
      print("basic butterfly: N=%, B=%, b=%: twiddle accesses: ", N, B, b);
      for 0..7 print("%, ", (it*b));
      print("\n");
      */

      //for 0..7  w[it] = unit_polar(-TAU*it*b/N); // Slow version


      // This works but we can do better, can't we
      // (With the version storing half of the twiddles and negating for the 
      // second half)
      /*
      for 0..7 {
        twiddle_index := it*b;
        w[it] = twiddles[twiddle_index & (N_div_2 - 1)];
        w[it] = w[it] * ifx twiddle_index >= N_div_2 then -1.0 else 1.0;

        //if (it <= 3 && twiddle_index > N_div_2) print("NOTE: it = %, twiddle_index = %, N_div_2 = %\n", it, twiddle_index, N_div_2);
      }
      */
      /*
      for 0..3 {
        twiddle_index := it*b;
        w[it] = twiddles[twiddle_index & (N_div_2 - 1)];
        w[it + 4] = -1.0 * twiddles[(twiddle_index+4) & (N_div_2 - 1)];
      }
      */


      // Dumber indexing strategy:
      for 0..7  w[it] = twiddles[B*it + b];

      for 0..7  i[it] = in[iarr + it*B];              // Gather inputs 

      for 1..7  t[it] = w[it] * i[it];             // Multiply twiddles

      // These constants multiply certain twiddles for the final output
      j :: Complex.{0.0, 1.0};
      w1_8 :: #run unit_polar(-TAU*1/8);    // sqrt(2) - sqrt(2)*j

      out[iarr + 0*B] = i[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
      out[iarr + 1*B] = i[0] + w1_8*t[1] - j*t[2] - j*w1_8*t[3] - t[4] - w1_8*t[5] + j*t[6] + j*w1_8*t[7];
      out[iarr + 2*B] = i[0] - j*t[1] - t[2] + j*t[3] + t[4] - j*t[5] - t[6] + j*t[7];
      out[iarr + 3*B] = i[0] - j*w1_8*t[1] + j*t[2] + w1_8*t[3] - t[4] + j*w1_8*t[5] - j*t[6] - w1_8*t[7];
      out[iarr + 4*B] = i[0] - t[1] + t[2] - t[3] + t[4] - t[5] + t[6] - t[7];
      out[iarr + 5*B] = i[0] - w1_8*t[1] - j*t[2] + j*w1_8*t[3] - t[4] + w1_8*t[5] + j*t[6] - j*w1_8*t[7];
      out[iarr + 6*B] = i[0] + j*t[1] - t[2] - j*t[3] + t[4] + j*t[5] - t[6] - j*t[7];
      out[iarr + 7*B] = i[0] + j*w1_8*t[1] + j*t[2] - w1_8*t[3] - t[4] - j*w1_8*t[5] - j*t[6] + w1_8*t[7];

    }
  }

  if !forward {
    for 0..fft_data.max_complex_transform_size-1  out[it] = out[it] * (1.0/N);
  }
}



// Uncomment this to print twiddles at compiletime if you want
// (useful when writing/debugging butterfly functions)
//#run print_hardcoded_twiddles_for_radix(8, forward=true);
//#run print_hardcoded_twiddles_for_radix(8, forward=false);

// (Side note: with Jai's powerful compiletime features we could use a similar function
// to actually generate the hardcoded butterfly as Code - this would be cool but seems
// hard to integrate with the assembly butterflies we need if we want to actually go fast.)


get_sub_array :: (array: [] $T, index: int, count: int) -> [] T {
  result: [] T;
  result.data = array.data + index;
  result.count = count;
  return result;
}

//unit_polar :: (theta: float) -> Complex { return .{ cos(theta), sin(theta) }; }

operator / :: inline (a: Complex, b: Complex) -> Complex {
  norm_b := length(b); 
  return .{
    (a.x*b.x + a.y*b.y) / norm_b,
    (b.x*a.y - a.x*b.y) / norm_b
  };
}

#scope_file

conj :: (z: Complex) -> Complex { return .{ z.x, -z.y }; }
#import,file "../module.jai";

#import "Math";
#import "Basic";
