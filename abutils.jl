module abutils

using PyPlot
using Distributions
using LinearAlgebra
using Formatting
using Clustering

macro noopwhen(condition, expression)
    quote
        if !($condition)
            $expression
        end
    end |> esc
end


# ========= MATH / ARRAYS ===================================
unpack_arr(x::AbstractArray) = [x[:,i] for i in range(1,stop=size(x,2))]
hstack(x) = reduce(hcat, x)
vstack(x) = reduce(vcat, x)


fastexp(x::Float64) = ccall((:exp, :libm), Float64, (Float64,), x)  # slightly faster (~75% of speed) than stdlib
fastlog(x::Float64) = ccall((:log, :libm), Float64, (Float64,), x)  # slightly faster (~75% of speed) than stdlib


eye(d) = Matrix(I, d, d)

function softmax2(logp; dims=2)
    p = exp.(logp .- maximum(logp, dims=dims))
    p ./= sum(p, dims=dims)
    return p
end


function sq_diff_matrix(X, Y)
    """
    Constructs (x_i - y_j)^T (x_i - y_j) matrix where
    X = Array{Float}(n_x, d)
    Y = Array{Float}(n_y, d)
    return: np.array(n_x, n_y)
    """
    normsq_x = sum(X.^2, dims=2)
    normsq_y = sum(Y.^2, dims=2)
    out = normsq_x .+ normsq_y'
    out -= 2*X * Y'
    return out
end
# ==================================================================================

function num_grad(fn, X, h=1e-8; verbose=true)
    """
    Calculate finite differences gradient of function fn evaluated at numpy
    array X. Not calculating central diff to improve speed and because some
    authors disagree about benefits.

    Most of the code here is really to deal with weird sizes of inputs or
    outputs. If scalar input and multi-dim output or vice versa, we return
    gradient in the shape of the multi-dim input or output. However, if both
    are multi-dimensional then we return as n_output vs n_input matrix
    where both input/outputs have been vectorised if necessary.
    """
    shp = size(X)
    resize_x = ndims(X) > 1
    rm_xdim = isa(X, Real)
    n = length(X)

    f_x = fn(X)
    if isa(f_x, Real)
        im_f_shp = 0
        resize_y = false
    else
        im_f_shp = size(f_x)
        resize_y = !(ndims(f_x) <= 2 && any(im_f_shp .== 1))
        @assert ndims(f_x) <= 2 "image of fn is tensor. Not supported."
    end
        
    m = Int64(prod(max.(im_f_shp, 1)))

    X = X[:]
    g = zeros(m, n)
    for ii in range(1, stop=n)
        Xplus = convert(Array{Float64}, copy(X))
        Xplus[ii] += h
        Xplus = reshape(Xplus, shp)
        grad = (fn(Xplus) - f_x) ./ h
        if ndims(grad) >= 1
            grad = ndims(grad) > 1 ? grad[:] : grad
            g[:, ii] = grad
        else
            g[ii] = grad
        end
    end
    
    verbose && resize_x && resize_y &&
        println("WARNING: Returning gradient as matrix size n(fn output) x n(variables)")

    if rm_xdim && size(g,2) == 1
        g = g[:]
    elseif resize_x && !any(im_f_shp .> 1)
        g = reshape(g, shp)
    elseif resize_y && !any(shp .> 1)
        g = reshape(g, im_f_shp)
    end

    return g
end


function num_grad_spec(fn, X, cart_ix, h=1e-8; verbose=true)
    """
    As per num_grad, but specifying a Cartesian Index ('cart_ix').
    Thus a n(fn output) x 1 array will always be returned. At
    some point it makes sense to roll this into my base numgrad, but
    I don't want to yet as it complicates things.
    """
    shp = size(X)
    resize_x = ndims(X) > 1
    rm_xdim = isa(X, Real)
    n = length(X)

    f_x = fn(X)
    if isa(f_x, Real)
        im_f_shp = 0
        resize_y = false
    else
        im_f_shp = size(f_x)
        resize_y = !(ndims(f_x) <= 2 && any(im_f_shp .== 1))
        @assert ndims(f_x) <= 2 "image of fn is tensor. Not supported."
    end
        
    m = Int64(prod(max.(im_f_shp, 1)))

    g = zeros(m, 1)
    Xplus = convert(Array{Float64}, copy(X))
    Xplus[cart_ix] += h
    Xplus = reshape(Xplus, shp)
    g = (fn(Xplus) - f_x) ./ h

    return g
end

# ==================================================================================


function multinomial_indices(n::Int, p::Vector{Float64})
    # adapted from Distributions.jl/src/samplers/multinomial.jl
    
    k = length(p)
    rp = 1.0  # remaining total probability
    i = 0
    km1 = k - 1
    x = zeros(Int32, n)
    op_ix = 1
    
    while i < km1 && n > 0
        i += 1
        @inbounds pi = p[i]
        if pi < rp            
            xi = rand(Binomial(n, pi / rp))
            x[op_ix:(op_ix+xi-1)] .= i
            op_ix += xi
            n -= xi
            rp -= pi
        else 
            # In this case, we don't even have to sample
            # from Binomial. Just assign remaining counts
            # to xi. 
            x[op_ix:(op_ix+n-1)] .= i
            n = 0
        end
    end

    if i == km1
        x[op_ix:end] .= i+1
    end

    return x  
end


### MULTINOMIAL SAMPLING FROM A LOG PROBABILITY VECTOR
function smp_from_logprob(n_samples::Int, logp::Vector{Float64})
    p = exp.(logp .- maximum(logp))
    p /= sum(p)
    return multinomial_indices(n_samples, p)
end   


# ==================================================================================

function gaussian_2D_level_curve(mu::Array{Float64,1}, sigma::Array{Float64, 2}, alpha=2, ncoods=100)
    @assert size(mu) == (2,) "mu must be vector in R^2"
    @assert size(sigma) == (2, 2) "sigma must be 2x2 array"

    U, S, V = svd(sigma)

    sd = sqrt.(S)
    coods = range(0, stop=2*pi, length=ncoods)
    coods = hstack((sd[1] * cos.(coods), sd[2] * sin.(coods)))' * alpha
    
    coods = (V' * coods)' # project onto basis of ellipse
    coods = coods .+ mu' # add mean
    return coods
end

function subplot_gridsize(num)
    poss = [[x,Int(ceil(num/x))] for x in range(1,stop=Int(floor(sqrt(num)))+1)]
    choice = findmin([sum(x) for x in poss])[2]  # argmin
    return sort(poss[choice])
end

function scatter_arrays(xs...)
    n = length(xs)
    sz = subplot_gridsize(n)
    f, axs = PyPlot.subplots(sz..., figsize=(5 + (sz[2]>1), sz[1]*3))
    if n == 1   # axs is not an array!
        axs[:scatter](unpack_arr(xs[1])...)
        return
    else
        for i in eachindex(xs)
            ax = axs[i]; x = xs[i]
            ax[:scatter](unpack_arr(x)...)
        end
    end
end


_cmapc = [ 0.12156863  0.46666667  0.70588235  1.     ;
           1.          0.49803922  0.05490196  1.        ;
           0.17254902  0.62745098  0.17254902  1.        ;
           0.83921569  0.15294118  0.15686275  1.        ]
tab10(i, a) = begin tmp = _cmapc[i, :]; tmp[end] = a; return tmp end



end