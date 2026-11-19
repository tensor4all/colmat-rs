// System V x86-64: rdi/rsi point to packed A/B, rdx to a contiguous 8x6
// product tile, rcx is the depth (a positive multiple of four), r8 selects
// accumulation. Each pair of YMM accumulators holds one column of the tile.
.irp reg,0,1,2,3,4,5,6,7,8,9,10,11
    vxorpd ymm\reg, ymm\reg, ymm\reg
.endr
.macro strassen_step ao, bo
vmovupd ymm12, [rdi+\ao+0]
vmovupd ymm13, [rdi+\ao+32]
vbroadcastsd ymm14, [rsi+\bo+0]
vbroadcastsd ymm15, [rsi+\bo+8]
vfmadd231pd ymm0, ymm12, ymm14
vfmadd231pd ymm1, ymm13, ymm14
vbroadcastsd ymm14, [rsi+\bo+16]
vfmadd231pd ymm2, ymm12, ymm15
vfmadd231pd ymm3, ymm13, ymm15
vbroadcastsd ymm15, [rsi+\bo+24]
vfmadd231pd ymm4, ymm12, ymm14
vfmadd231pd ymm5, ymm13, ymm14
vbroadcastsd ymm14, [rsi+\bo+32]
vfmadd231pd ymm6, ymm12, ymm15
vfmadd231pd ymm7, ymm13, ymm15
vbroadcastsd ymm15, [rsi+\bo+40]
vfmadd231pd ymm8, ymm12, ymm14
vfmadd231pd ymm9, ymm13, ymm14
vfmadd231pd ymm10, ymm12, ymm15
vfmadd231pd ymm11, ymm13, ymm15
.endm
shr rcx, 2
.p2align 5
2:
    // Warm two lines of the streamed A panel ahead of this unrolled group.
    prefetcht0 [rdi+1536]
    prefetcht0 [rdi+1600]
    strassen_step 0, 0
    strassen_step 64, 48
    strassen_step 128, 96
    strassen_step 192, 144
    add rdi, 256
    add rsi, 192
    dec rcx
    jnz 2b
.purgem strassen_step
test r8, r8
jz 3f
.irp reg,0,1,2,3,4,5,6,7,8,9,10,11
    vaddpd ymm\reg, ymm\reg, [rdx+32*\reg]
.endr
3:
.irp reg,0,1,2,3,4,5,6,7,8,9,10,11
    vmovupd [rdx+32*\reg], ymm\reg
.endr
vzeroupper
ret
