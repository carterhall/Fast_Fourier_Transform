//
// @NOTE: When using these functions, if you want to use the same register for 
// input and output, the shared register MUST go into the first input slot, NOT
// the second:
//   mul(v1, v1, v2);  // Good
//   mul(v1, v2, v1);  // Wrong!!!
//
// On AVX+, there is no difference, but with normal SSE instructions, we have to
// choose one of the two arguments to move into the output register first, and 
// that will cause it to be overwritten if you use the wrong calling order.
//

// This makes it clearer that we might call a smaller one even if we support AVX etc, and 
// that at sizes 64 and 128, we may or may not be able to use SSE3 instructions.
// It is ordered by size, so that some of the checks can be less-than, greater-than, etc.
Mode :: enum { NO_SIMD; SSE_64; SSE3_64; SSE_128; SSE3_128; AVX; AVX512; }

complex_mul :: (out: __reg, in0: __reg, in1: __reg, $mode: Mode) #expand {
  _one := 1.0; // @Speed see if we can cast this to int to save regs
  one := *_one;

  #if mode == .AVX512  #asm AVX512F {
    v1: vec; v2: vec; v0: vec;
    movsldup.z v0, in0;  // Complex multiplication, per Intel optimization manual
    movaps.z   v1, in1;
    mulps.z    v0, v0, v1;
    shufps.z   v1, v1, v1, 0xb1;
    movshdup.z v2, in0;
    mulps.z    v2, v2, v1;

    broadcastss.z v1, [one];
    fmaddsub132ps v0, v2, v1;
    movaps.z      out, v0;
  }
  else #if mode == .AVX  #asm AVX {
    v1: vec; v2: vec; v0: vec;
    movsldup.y v0, in0;  // Complex multiplication, per Intel optimization manual
    movaps.y   v1, in1;
    mulps.y    v0, v0, v1;
    shufps.y   v1, v1, v1, 0xb1;
    movshdup.y v2, in0;
    mulps.y    v2, v2, v1;
    addsubps.y v0, v0, v2;
    movaps.y   out, v0;
  }
  else #if mode == .SSE3_128 || mode == .SSE3_64  #asm SSE3 {
    // Take care here to avoid overwriting the output if it is also an input!
    v1: vec; v2: vec; v0: vec;
    movsldup.x v0, in0;  // SSE3
    movaps.x   v1, in1;
    mulps.x    v0, v1;
    shufps.x   v1, v1, 0xb1;
    movshdup.x v2, in0;   // SSE3
    mulps.x    v2, v1;
    addsubps.x v0, v2;   // SSE3
    movaps.x   out, v0;
  }
  else #if mode == .SSE_128 || mode == .SSE_64  {
    // @Speed
    // This is obviously silly, and not fast, but it seems like there's not a very fast
    // way to do complex multiplication before the SSE3 instructions.

    // Almost nobody will hit this path anyway...

    __in0, __in1, __out: [2] Complex;
    _in0, _in1, _out := __in0.data, __in1.data, __out.data;
    #asm SSE {
      movaps.x [_in0], in0;
      movaps.x [_in1], in1;
    }
    __out[0] = __in0[0]*__in1[0];
    __out[1] = __in0[1]*__in1[1];
    #asm SSE {
      movaps.x out, [_out]; 
    }
  }
}

complex_negate :: (reg: __reg, $mode: Mode) #expand {
  negative_one := -1.0;
  n1 := *negative_one;

  #if mode == .AVX512  #asm AVX512F {
    broadcastss.z v1:, [n1];
    mulps         reg, v1;
  }
  else #if mode == .AVX  #asm AVX {
    broadcastss.y v1:, [n1];
    mulps         reg, v1;
  }
  else #asm SSE {
    movss  v1:, [n1]; 
    shufps v1, v1, 0;
    mulps  reg, v1;
  }
}

complex_add :: (out: __reg, in1: __reg, in2: __reg, $mode: Mode) #expand {
  #if      mode == .AVX512  #asm AVX512F  { addps.z  out, in1, in2; } 
  else #if mode == .AVX     #asm AVX      { addps.y  out, in1, in2; } 
  else                      #asm SSE      { movaps.x out, in1;  addps.x out, in2; }
}

complex_sub :: (out: __reg, in1: __reg, in2: __reg, $mode: Mode) #expand {
  #if      mode == .AVX512  #asm AVX512F  { subps.z  out, in1, in2; } 
  else #if mode == .AVX     #asm AVX      { subps.y  out, in1, in2; } 
  else                      #asm SSE      { movaps.x out, in1;  subps.x out, in2; }
}

// If load/store/mov are called in a 64-bit mode, they pretend they're working on a 
// double-precision scalar. Otherwise, they use the full width of the SIMD featureset.
complex_load :: (reg: __reg, ptr: *Complex, $mode: Mode) #expand {
  #if      mode == .AVX512   #asm AVX512F  { movups.z reg, [ptr]; } 
  else #if mode == .AVX      #asm AVX      { movups.y reg, [ptr]; } 
  else #if mode >= .SSE_128  #asm SSE      { movups.x reg, [ptr]; } 
  else                       #asm SSE      { movsd    reg, [ptr]; }
}

complex_store :: (ptr: *Complex, reg: __reg, $mode: Mode) #expand {
  #if      mode == .AVX512   #asm AVX512F  { movups.z [ptr], reg; } 
  else #if mode == .AVX      #asm AVX      { movups.y [ptr], reg; } 
  else #if mode >= .SSE_128  #asm SSE      { movups.x [ptr], reg; } 
  else                       #asm SSE      { movsd    [ptr], reg; }
}

complex_mov :: (out: __reg, in: __reg, $mode: Mode) #expand {
  #if      mode == .AVX512   #asm AVX512F  { movaps.z out, in; } 
  else #if mode == .AVX      #asm AVX      { movaps.y out, in; } 
  else #if mode >= .SSE_128  #asm SSE      { movaps.x out, in; } 
  else                       #asm SSE      { movsd    out, in; }
}
