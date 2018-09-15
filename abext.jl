module abext

using Flux
using Flux.Tracker: @grad
using LinearAlgebra: dot
using NNlib: softmax
using Base.Threads: @threads
import StatsFuns: logsumexp

using PyPlot
include("./abutils.jl")



## THREADED MAP REDUCE. I.E I want threaded loop of a reduction.
function tmapreduce(f::Function, op, v0, itr)
    @assert length(itr) > 0
    mutex = Threads.Mutex()
    output = deepcopy(v0) # make a deepcopy of starting value
    poppable_itr = Vector(deepcopy(itr)) # convert to set to be able to pop
    # array of input arguments to f, to store input for each thread
    inputs = Array{typeof(itr[1])}(undef, Threads.nthreads()) # deprecated soon?
    @threads for i in eachindex(itr)
        lock(mutex)
        inputs[Threads.threadid()] = pop!(poppable_itr)
        unlock(mutex)
        loop_output = f(inputs[Threads.threadid()])
        lock(mutex)
        output = op(output, loop_output)
        unlock(mutex)
    end
    return output
end


# Binary Cross Entropy

# GRADIENT IS APPROX THE SAME REGARDLESS OF THREADING: THESE ARE BLAS (I THINK!)
∇P_BCE(X, P) = - (X - P) ./ ((P .+ eps()) .* (1 .- P .+ eps()))
∇X_BCE(P) =  - (log.(P .+ eps()) - log.(1 .- P .+ eps()))


function BCE_unthreaded(X, P)
	@assert size(X) == size(P)
    bce = 0.0
    bce_fn(x, p) = dot(x, log.(p .+ eps())) + dot(1 .- x, log.(1 .- p .+ eps()))
    @views for i in 1:size(X,2)
        bce -= bce_fn(X[:,i], P[:,i])
    end
    return bce
end

BCE_unthreaded(X::TrackedArray, P::TrackedArray) = Tracker.track(BCE, X, P)
BCE_unthreaded(X::AbstractArray, P::TrackedArray) = Tracker.track(BCE, X, P)
BCE_unthreaded(X::TrackedArray, P::AbstractArray) = Tracker.track(BCE, X, P)

@grad function BCE_unthreaded(X, P)
    return BCE_unthreaded(Tracker.data(X), Tracker.data(P)), Δ -> (Δ * ∇X_BCE(P), Δ * ∇P_BCE(X, P))
end


# Binary Cross Entropy
function BCE(X, P)
	@assert size(X) == size(P)
    function col_bce(i)
    	@views x_i = X[:,i]
    	@views p_i = P[:,i]
    	return -dot(x_i, log.(p_i .+ eps())) + dot(1 .- x_i, log.(1 .- p_i .+ eps()))
    end

    return tmapreduce(col_bce, +, 0.0, 1:size(X,2))
end

BCE(X::TrackedArray, P::TrackedArray) = Tracker.track(BCE, X, P)
BCE(X::AbstractArray, P::TrackedArray) = Tracker.track(BCE, X, P)
BCE(X::TrackedArray, P::AbstractArray) = Tracker.track(BCE, X, P)

∇P_BCE(X, P) = - (X - P) ./ ((P .+ eps()) .* (1 .- P .+ eps()))
∇X_BCE(P) =  - (log.(P .+ eps()) - log.(1 .- P .+ eps()))

@grad function BCE(X, P)
    return BCE(Tracker.data(X), Tracker.data(P)), Δ -> (Δ * ∇X_BCE(P), Δ * ∇P_BCE(X, P))
end

binary_cross_entropy = BCE   # (alias)



# Weighted Binary Cross Entropy

# Binary Cross Entropy
function BCE_weightgrad(X, P, W)
	@assert size(X) == size(P)
    bce = 0.0
    bce_fn(x, p) = dot(x, log.(p .+ eps())) + dot(1 .- x, log.(1 .- p .+ eps()))
    @views for i in 1:size(X,2)
        bce -= bce_fn(X[:,i], P[:,i])
    end
    return bce
end

function ∇P_BCEW(X, P, W) 
	return - ((X - P) ./ ((P .+ eps()) .* (1 .- P .+ eps()))) .* W
end

BCE_weightgrad(X, P, W) = BCE(X, P)
BCE_weightgrad(X::TrackedArray, P::TrackedArray, W::AbstractArray) = Tracker.track(BCE_weightgrad, X, P, W)
BCE_weightgrad(X::AbstractArray, P::TrackedArray, W::AbstractArray) = Tracker.track(BCE_weightgrad, X, P, W)
BCE_weightgrad(X::TrackedArray, P::AbstractArray, W::AbstractArray) = Tracker.track(BCE_weightgrad, X, P, W)


@grad function BCE_weightgrad(X, P, W)
	@assert size(W) == (1, size(X, 2))
    return BCE(Tracker.data(X), Tracker.data(P)), Δ -> (Δ * ∇X_BCE(P), Δ * ∇P_BCEW(X, P, W), zeros(size(W)))
end

binary_cross_entropy = BCE   # (alias)


# Binary Cross Entropy
function BCE(X, P)
	@assert size(X) == size(P)
    bce = 0.0
    bce_fn(x, p) = dot(x, log.(p .+ eps())) + dot(1 .- x, log.(1 .- p .+ eps()))
    @views for i in 1:size(X,2)
        bce -= bce_fn(X[:,i], P[:,i])
    end
    return bce
end

BCE(X::TrackedArray, P::TrackedArray) = Tracker.track(BCE, X, P)
BCE(X::AbstractArray, P::TrackedArray) = Tracker.track(BCE, X, P)
BCE(X::TrackedArray, P::AbstractArray) = Tracker.track(BCE, X, P)

∇P_BCE(X, P) = - (X - P) ./ ((P .+ eps()) .* (1 .- P .+ eps()))
∇X_BCE(P) =  - (log.(P .+ eps()) - log.(1 .- P .+ eps()))

@grad function BCE(X, P)
    return BCE(Tracker.data(X), Tracker.data(P)), Δ -> (Δ * ∇X_BCE(P), Δ * ∇P_BCE(X, P))
end



function zero_grad!(ps)
    for p in ps
        p.tracker.grad .= 0
    end
end


# REQUIRED TO APPROACH NUMPY SPEEDS (ALTHOUGH STILL APPROX 2x).
# Need VML to accelerate this, but this only just hit 0.7
function threadedlog(x::AbstractArray)
  out = similar(x)
  Threads.@threads for i in eachindex(x)
    out[i]=log(x[i])
  end
  return out
end

function threadedexp(x::AbstractArray)
  out = similar(x)
  Threads.@threads for i in eachindex(x)
    out[i]=exp(x[i])
  end
  return out
end
end