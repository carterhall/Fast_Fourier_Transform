mode_to_size :: (mode: Mode) -> int {
  if mode == .AVX512   return 8;
  if mode == .AVX      return 4;
  if mode == .SSE3_128 return 2;
  if mode == .SSE_128  return 2;
  return 1;
}
FFT :: #import,file "../module.jai";

run_tests_at_simd_mode :: ($mode: Mode) {
  // Create the test plan and declare all the registers we need at a given support level
  #if mode == .AVX512 {
    #asm AVX512F { yy: vec; aa: vec; bb: vec; }
  } else #if mode == .AVX {
    #asm AVX     { yy: vec; aa: vec; bb: vec; }
  } else #if mode >= .SSE_64  {
    #asm SSE     { yy: vec; aa: vec; bb: vec; }
  } else {
    #run print("Error: no SIMD support detected!\n");
    #assert(0);
  }

  N :: #run mode_to_size(mode);

  // Addition 
  {
    print("\n==== SIMD mode = %, testing addition...\n", mode);

    a, b, y, y_ref: [N] Complex;

    initialize_array_with_random_values(a, 0);
    initialize_array_with_random_values(b, 1);
    initialize_array_with_random_values(y, 2);

    complex_load(aa, a.data, mode);
    complex_load(bb, b.data, mode);
    complex_add(yy, aa, bb, mode);
    complex_store(y.data, yy, mode);

    print("  a = %\n", a);
    print("  b = %\n", b);
    print("  y =     %\n", y);

    for 0..N-1  y_ref[it] = a[it] + b[it];
    print("  y_ref = %\n", y_ref);

    for 0..N-1  assert(y_ref[it] == y[it]);
  }

  // Addition with first arg as both input and output
  {
    print("\n==== SIMD mode = %, testing addition with shared i/o register...\n", mode);

    a, y, y_ref: [N] Complex;

    initialize_array_with_random_values(a, 0);
    initialize_array_with_random_values(y, 2);

    print("  a = %\n", a);
    print("  y =     %\n", y);

    for 0..N-1  y_ref[it] = a[it] + y[it];
    print("  y_ref = %\n", y_ref);

    complex_load(aa, a.data, mode);
    complex_load(yy, y.data, mode);
    complex_add(yy, yy, aa, mode);
    //complex_add(yy, aa, yy, mode);  // This one would be wrong!!
    complex_store(y.data, yy, mode);

    for 0..N-1  assert(y_ref[it] == y[it]);
  }

  // Subtraction 
  {
    print("\n==== SIMD mode = %, testing subtraction...\n", mode);

    a, b, y, y_ref: [N] Complex;

    initialize_array_with_random_values(a, 0);
    initialize_array_with_random_values(b, 1);
    initialize_array_with_random_values(y, 2);

    complex_load(aa, a.data, mode);
    complex_load(bb, b.data, mode);
    complex_sub(yy, aa, bb, mode);
    complex_store(y.data, yy, mode);

    print("  a = %\n", a);
    print("  b = %\n", b);
    print("  y =     %\n", y);

    for 0..N-1  y_ref[it] = a[it] - b[it];
    print("  y_ref = %\n", y_ref);

    for 0..N-1  assert(y_ref[it] == y[it]);
  }

  // Multiplication
  {
    print("\n==== SIMD mode = %, testing complex multiplication...\n", mode);

    a, b, y, y_ref: [N] Complex;

    initialize_array_with_random_values(a, 0);
    initialize_array_with_random_values(b, 1);
    initialize_array_with_random_values(y, 2);

    complex_load(aa, a.data, mode);
    complex_load(bb, b.data, mode);
    complex_mul(yy, aa, bb, mode);
    complex_store(y.data, yy, mode);

    print("  a = %\n", a);
    print("  b = %\n", b);
    print("  y =     %\n", y);

    for 0..N-1  y_ref[it] = a[it] * b[it];
    print("  y_ref = %\n", y_ref);

    for 0..N-1  assert(y_ref[it] == y[it]);
  }

  // Multiplication with shared i/o register
  {
    print("\n==== SIMD mode = %, testing complex multiplication with shared i/o register...\n", mode);

    a, y, y_ref: [N] Complex;

    initialize_array_with_random_values(a, 0);
    initialize_array_with_random_values(y, 2);

    print("  a = %\n", a);
    print("  y =     %\n", y);

    for 0..N-1  y_ref[it] = a[it] * y[it];
    print("  y_ref = %\n", y_ref);

    complex_load(aa, a.data, mode);
    complex_load(yy, y.data, mode);
    //complex_mul(yy, aa, yy, mode);
    complex_mul(yy, yy, aa, mode);
    complex_store(y.data, yy, mode);

    for 0..N-1  assert(y_ref[it] == y[it]);
  }
}

main :: () {
  simd_support := FFT.check_simd_support(); 

  if simd_support >= .SSE {
    run_tests_at_simd_mode(.SSE_64); 
    run_tests_at_simd_mode(.SSE_128);
  }
  if simd_support >= .SSE3 {
    run_tests_at_simd_mode(.SSE3_64); 
    run_tests_at_simd_mode(.SSE3_128);
  }
  if simd_support >= .AVX {
    run_tests_at_simd_mode(.AVX);
  }
  if simd_support >= .AVX512 {
    run_tests_at_simd_mode(.AVX512);
  }

  print("SIMD helper test completed without errors.\n");
}

#load "../simd_helpers.jai";

#import "Math";
#import "Basic";
Random :: #import "Random";

// Pasta from test_fft
initialize_array_with_random_values :: (arr: [] $T, seed: u64 = 1337) {
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

