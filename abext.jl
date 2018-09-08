module abext

using Flux
using Flux.Tracker: @grad
using LinearAlgebra: dot
using NNlib: softmax
import StatsFuns: logsumexp

using PyPlot
include("./abutils.jl")

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

end