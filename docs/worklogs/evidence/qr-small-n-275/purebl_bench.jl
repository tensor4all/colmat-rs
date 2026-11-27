using PureBLAS, LinearAlgebra, Statistics
println("threads=",Threads.nthreads()," PureBLAS ",pkgversion(PureBLAS))
function bench(f, A; batches=15, inner=200)
    B=copy(A); f(B); ts=Float64[]
    for _ in 1:batches
        tot=0.0
        for _ in 1:inner
            copyto!(B,A); t=time_ns(); f(B); tot+=time_ns()-t
        end
        push!(ts, tot/inner/1e3)
    end
    (median(ts), minimum(ts))
end
for n in (96,128)
    A=randn(n,n)
    m,mn=bench(PureBLAS.geqrf!,A); println("PureBLAS geqrf! n=$n median $(round(m,digits=1)) us min $(round(mn,digits=1)) us (15x200)")
    m,mn=bench(B->LAPACK.geqrf!(B),A); println("OpenBLAS geqrf! n=$n median $(round(m,digits=1)) us min $(round(mn,digits=1)) us (15x200)")
end
