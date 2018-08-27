using PyPlot
using Distributions
using LinearAlgebra
using Formatting
using Flux
using Flux: Tracker
using StatsBase
using Clustering
using Parameters
using Random
using Flux.Tracker: TrackedReal, track, @grad


splat(x::AbstractArray) = [x[:,i] for i in range(1,stop=size(x,2))]
fastexp(x::Float64) = ccall((:exp, :libm), Float64, (Float64,), x)  # slightly faster (~75% of speed) than stdlib
fastlog(x::Float64) = ccall((:log, :libm), Float64, (Float64,), x)  # slightly faster (~75% of speed) than stdlib


function gaussian_2D_level_curve(mu::Array{Float64,1}, sigma::Array{Float64, 2}, alpha=2, ncoods=100)
    @assert size(mu) == (2,) "mu must be vector in R^2"
    @assert size(sigma) == (2, 2) "sigma must be 2x2 array"

    U, S, V = svd(sigma)

    sd = sqrt.(S)
    coods = range(0, stop=2*pi, length=ncoods)
    coods = reduce(hcat, (sd[1] * cos.(coods), sd[2] * sin.(coods)))' * alpha
    
    coods = (V' * coods)' # project onto basis of ellipse
    coods = coods .+ mu' # add mean
    return coods
end

function plot_is_vs_target(S, W; ax=Nothing, kwargs...)
    rgba_colors = zeros(size(S, 1), 4)
    rgba_colors[:, 3] .= 1.0   # blue
    rgba_colors[:, 4] = W/maximum(W)   # alpha
#     print(rgba_colors)
    ax = ax==Nothing ? gca() : ax
#     plot_level_curves_all(mus, UTs; ax=ax)
    ax[:scatter](splat(S)..., c=rgba_colors)
end

eye(d) = Matrix(I, d, d)
logical_not(x) = x .== false

macro noopwhen(condition, expression)
    quote
        if !($condition)
            $expression
        end
    end |> esc
end

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



eff_ss(W) = 1/sum((W./sum(W)).^2)
weight_perp(W) = let w=W/sum(W); -sum(w.* log.(w))/log(length(W)) end



# For Automatic Differentiation / GMM
# ====================================================

function logsumexp(X::AbstractArray{T}) where {T<:Real}
    isempty(X) && return log(zero(T))
    u = maximum(X)
    isfinite(u) || return float(u)
    let u=u # avoid https://github.com/JuliaLang/julia/issues/15276
        u + log.(sum(x -> exp.(x-u), X))
    end
end

function logsumexprows(X::AbstractArray{T}) where {T<:Real}
    n = size(X,1)
    out = zeros(n)
    for i = 1:n
        out[i] = logsumexp(X[i,:])
    end
    return out
end   


function softmax2(logp; dims=2)
    p = exp.(logp .- maximum(logp, dims=dims))
    return p ./ sum(p, dims=dims)
end

# PLUG INTO FLUX
logsumexprows(X::TrackedArray) = Tracker.track(logsumexprows, X)

@grad function logsumexprows(X)
  return logsumexprows(X.data), Δ -> (Δ .* softmax2(Δ.*X.data),)
end

# ====================================================



function eta_t(t)
    delta = 0.1
    return delta * exp(-0.02 * t)
end



function gradient_importance_sample_gauss(epochs, n_samples, log_f; burnin=0, test=true, prior_std=10.)
    @assert burnin < epochs
    
    sqrtdelta = 1
    S = zeros(n_samples*epochs, 2)
    W = zeros(n_samples*epochs)
    
    # Make initial proposal from prior
    μ_init = [0 0]'
    x = randn(n_samples, 2) .* prior_std .+ μ_init'
    S[1:n_samples,:] = x
    
    # ... allow for resampling of all (in principle)
    W[1:n_samples] .= 1
    
    @noopwhen !test f, axs = PyPlot.subplots(11,3, figsize=(10,20))
    @noopwhen !test axs[1,1][:scatter](splat(x)...)
    @noopwhen !test plot_is_vs_target(S[1:n_samples,:], exp.(W[1:n_samples]), ax=axs[1,2])
    @noopwhen !test display(reduce(hcat, [S[1:n_samples,1], S[1:n_samples,2], W[1:n_samples], log_f(x)[1]]))
    
    # ===> GRIS LOOP <====
    for t = range(1, stop=epochs)
        
        nodisp = (!test || t>10)
        # sample previous particles for this epoch
        begin
            min_rng = 1 + max(0, n_samples*t-1200)
            max_rng = n_samples*max(1, t-1)
            ixs = smp_from_logprob(n_samples, W[min_rng:max_rng])  .+ min_rng .- 1
        end
        
        x = S[ixs,:]
        
        @noopwhen nodisp axs[t+1,1][:scatter](splat(x)...)
        
        # get gradient / Langevin proposal
        # --- calculate gradient ---
        η = eta_t(t)  
        x_track = param(x)
        lp = log_f(x_track)[1]
        Tracker.back!(sum(lp))
        
        c_mus = x .+ 0.0.*η .* x_track.grad
        # --------------------------
        
        # Sample from proposal
        xprime = c_mus .+ sqrtdelta*randn(n_samples, 2)
        @noopwhen nodisp axs[t+1,3][:scatter](splat(xprime)...)
        
        # calculate importance weight
        begin
            dist_metric = sq_diff_matrix(xprime, c_mus) ./ (2*sqrtdelta^2)
            lq_u = logsumexprows(-dist_metric) .- log(Float64(n_samples)) .- 0.5*2*log(2*pi*sqrtdelta^2)
            
            @noopwhen !test begin
                lq_u2 = logsumexprows(-dist_metric) .- log(Float64(n_samples)) .- 0.5*log(det(2*pi*sqrtdelta^2 * eye(2)))
                q_n = mapslices(x -> [mean([pdf(MvNormal(c_mus[i,:], sqrtdelta^2*eye(2)), x) 
                                      for i in 1:n_samples])], xprime, dims=2)
                print(unique(map(x->round(x, digits=5), exp.(lq_u2) ./exp.(lq_u))))   # ratio to exact quick calc
                println(unique(map(x->round(x, digits=5), q_n ./exp.(lq_u))))  # ratio to built-in Julia MVN
            end
                         
            w = log_f(xprime)[1] .- lq_u  # may be able to get rid of f_log_target if we resample as in Schuster
                                            # i.e. calculate in the gradient block in next iter.
        end
        
        # Store particles / weights
        S[1+n_samples*(t-1):n_samples*t, :] = xprime
        W[1+n_samples*(t-1):n_samples*t] = w
    end
    
    if burnin > 0
        S = S[n_samples*burnin+1:end, :]; W = W[n_samples*burnin+1:end]
    end
    
    return S, fastexp.(W)
end



@with_kw struct smcs_opt
    resample_every::Int64 = 1 
    sqrtdelta::Float64 =1.
    betas::Array{Float64} = [1.]
    test::Bool = false
    grad_delta::Float64 = 1.0
    burnin::Int64 = 0
    n_init::Int64 = 1000
    prior_std::Float64 = 10.
end


function smcs_grad(epochs, n_samples, log_f; opts=smcs_opt())
    @assert opts.burnin < epochs
    @assert isa(opts, smcs_opt)
    @assert epochs % opts.resample_every == 0 "epochs should be a multiple of the resampling points"
    
    @unpack test, resample_every, sqrtdelta, grad_delta = opts
    δ = grad_delta
    n_β = length(opts.betas)
    n_β > opts.burnin && println("WARNING: BURN IN SHORTER THAN ANNEALING TIME")
        
    S = zeros(n_samples*epochs, 2)
    W = zeros(n_samples*epochs)
    Wfinal = zeros(n_samples*epochs)
    
    # Make initial proposal from prior
    μ_init = [0 0]'
    x = randn(n_samples, 2) .* opts.prior_std .+ μ_init'
#     S[1:n_samples,:] = x
    
    @noopwhen !test f, axs = PyPlot.subplots(13,3, figsize=(10,22))
    @noopwhen !test axs[1,1][:scatter](splat(x)...)
    @noopwhen !test plot_is_vs_target(S[1:n_samples,:], exp.(W[1:n_samples]), ax=axs[1,2])
    @noopwhen !test display(reduce(hcat, [S[1:n_samples,1], S[1:n_samples,2], W[1:n_samples], log_f(x, opts.betas[1])[1]]))
    
    # ===> GRIS LOOP <====
    for t = range(1, stop=epochs)
        
        nodisp = (!test || t>12)
        β = opts.betas[min(t, n_β)]
        
        @noopwhen nodisp axs[t+1,1][:scatter](splat(x)...)
        
        # get gradient / Langevin proposal
        # --- calculate gradient ---
        η = eta_t(t)  
        x_track = param(x)
        lp = log_f(x_track, 1.)[1]   # also spits out just log_f in second retval (a.o.to beta mix)
        Tracker.back!(sum(lp))
        
        c_mus = x .+ δ.*η .* x_track.grad
        # --------------------------

        @noopwhen nodisp axs[t+1,2][:scatter](splat(c_mus)...)
        
        # Sample from proposal
        xprime = c_mus .+ sqrtdelta*randn(n_samples, 2)
        @noopwhen nodisp axs[t+1,3][:scatter](splat(xprime)...)
        
        # calculate importance weight
        begin
            dist_metric = sq_diff_matrix(xprime, c_mus) ./ (2*sqrtdelta^2)
            lq_u = logsumexprows(-dist_metric) .- log(Float64(n_samples)) .- 0.5*2*log(2*pi*sqrtdelta^2)
            
            @noopwhen true begin
                lq_u2 = logsumexprows(-dist_metric) .- log(Float64(n_samples)) .- 0.5*log(det(2*pi*sqrtdelta^2 * eye(2)))
                q_n = mapslices(x -> [mean([pdf(MvNormal(c_mus[i,:], sqrtdelta^2*eye(2)), x) 
                                      for i in 1:n_samples])], xprime, dims=2)
                print(unique(map(x->round(x, digits=5), exp.(lq_u2) ./exp.(lq_u))))   # ratio to exact quick calc
                println(unique(map(x->round(x, digits=5), q_n ./exp.(lq_u))))  # ratio to built-in Julia MVN
            end
            
            l_anneal, lp = log_f(xprime, β)
            w = l_anneal .- lq_u  # may be able to get rid of f_log_target if we resample as in Schuster
                                           # i.e. calculate in the gradient block in next iter.
        end
        
        # Store particles / weights & Resample (or just continue!)
        S[1+n_samples*(t-1):n_samples*t, :] = xprime
        W[1+n_samples*(t-1):n_samples*t] = w
        Wfinal[1+n_samples*(t-1):n_samples*t] = lp .- lq_u
        
        if t % resample_every == 0
            # RESAMPLE
            begin
                min_rng = 1 + max(0, n_samples*t-1200)
                max_rng = n_samples*max(1, t-1)
                ixs = smp_from_logprob(n_samples, W[min_rng:max_rng])  .+ min_rng .- 1
                @noopwhen !test print("RESAMPLE.." * sprintf1("%d", min_rng) * " " * sprintf1("%d", max_rng))
                x = S[ixs,:]
                @noopwhen !test println(" Done!")
            end
        else
            x = xprime
        end
    end
    
    if opts.burnin > 0
        ix_start = n_samples*opts.burnin+1
        S = S[ix_start:end, :]; W = Wfinal[ix_start:end]
    end
    
    return S, fastexp.(W)
end


# ===================================================================================================================
#    AMIS
# ===================================================================================================================

function gmm_llh(X, weights, pis, mus, sigmas)
    n, p = size(X)
    k = length(pis)
    thrsh_comp = 0.005
    inactive_ixs = pis[:] .< thrsh_comp
    
    P = zeros(n, k)
    for j = 1:k 
        P[:,j] = log_gauss_llh(X, mus[j,:], sigmas[:,:,j], 
            bypass=inactive_ixs[j]) .+ log(pis[j])
    end
    P .*= weights
    return logsumexprows(P)
end

function gmm_prior_llh(pis, mus, sigmas, pi_prior, mu_prior, cov_prior)
    d = size(cov_prior, 1)
    ν = pi_prior # alias
    k = length(pis)
    out = zeros(k)
    for j = 1:k
        out[j] += logpdf(MvNormal(mu_prior[j,:], sigmas[:,:,j]/ν[j]), mus[j,:])
        out[j] += -(ν[j] + d + 1)*logdet(sigmas[:,:,j])/2 
        out[j] += -ν[j]*sum(diag(cov_prior[:,:,j]*inv(sigmas[:,:,j])))/2
        out[j] += (ν[j] - 1)*log(pis[j])
    end
    return sum(out)
end


function log_gauss_llh(X, mu, sigma; bypass=false)
    if bypass 
        return -ones(size(X, 1))*Inf
    else
        retval = try _log_gauss_llh(X, mu, sigma)
            catch e
                return -ones(size(X, 1))*Inf
        	end
        return retval
    end
end
    
function _log_gauss_llh(X, mu, sigma)
    d = size(X,2)
#     invUT = Matrix(cholesky(inv(sigma)).U)
    invUT = inv(cholesky(sigma).L)
    Z = (X .- mu')*invUT'
    exponent = -0.5*sum(Z.^2, dims=2)
    lognormconst = -d*log(2*pi)/2 -0.5*logdet(sigma) #.+ sum(log.(diag(invUT)))
    return exponent .+ lognormconst
end

function gmm_custom(X, weights, pi_prior, mu_prior, cov_prior; max_iter=100, tol=1e-3, verbose=true)
    n, p = size(X)
    k = length(pi_prior)
    @assert size(weights) == (n,)
    @assert size(mu_prior) == (k, p)
    @assert size(cov_prior) == (p, p, k)
    pis = pi_prior/sum(pi_prior)
    mus = copy(mu_prior)
    sigmas = copy(cov_prior)
    
    weights = weights / mean(weights)   # diff. to Cappé et al. due to prior
    
    thrsh_comp = 0.005
    inactive_ixs = pi_prior[:] .< thrsh_comp
    pi_prior = copy(pi_prior)
    Ns = zeros(6)
    
    for i in range(1, stop=max_iter)
        # E-step
        rs = reduce(hcat, map(j -> log_gauss_llh(X, mus[j,:], sigmas[:,:,j], bypass=inactive_ixs[j]), 1:k))
        try
            rs .+= log.(pis)[:]'
            catch e
            display(rs)
            display(log.(pis))
            rethrow(e)
        end
        
        rs = softmax2(rs, dims=2)
        # reweight according to importance weights (see Adaptive IS in General Mix. Cappé et al. 2008)
        rs .*= weights
        
        # M-step
        Ns = sum(rs, dims=1)
        inactive_ixs = Ns[:] .< 1
        active_ixs = logical_not(inactive_ixs)  # can't find a native NOT for BitArrays in Julia
        if any(inactive_ixs)
            pis[inactive_ixs] .= 0.
            pi_prior[inactive_ixs] .= 0.
        end
        pis = Ns[:] + pi_prior[:]
        
        pis /= sum(pis)
        
        _mus = reduce(vcat, map(j -> sum(X .* rs[:,j], dims=1) .+ pi_prior[j]*mu_prior[j,:]', findall(active_ixs)))
        _mus ./= vec(Ns[active_ixs] + pi_prior[active_ixs])
        mus[active_ixs,:] = _mus
        
        for j in findall(active_ixs)
            Δx = X .- mus[j, :]'
            Δμ = (mus[j,:] - mu_prior[j,:])'
            sigmas[:,:,j] = (Δx.*rs[:,j])'Δx + pi_prior[j]*(Δμ'Δμ + cov_prior[:,:,j])
            sigmas[:,:,j] ./= (Ns[j] + pi_prior[j] + p + 2)
            sigmas[:,:,j] = (sigmas[:,:,j] + sigmas[:,:,j]')/2 + eye(2)*1e-6
        end

    end
    
    return pis, mus, sigmas
end

function sample_from_gmm(n, pis, mus, covs; shuffle=true)
    k, p = size(mus)
    Ns = rand(Multinomial(n, pis[:]))
    active_ixs = findall(Ns[:] .>= 1)
    
    ixs = hcat(vcat(1, 1 .+ cumsum(Ns[1:end-1], dims=1)), cumsum(Ns, dims=1))
    out = zeros(n, p)
    for j=active_ixs
        out[ixs[j,1]:ixs[j,2],:] = rand(MvNormal(mus[j,:], covs[:,:,j]), Ns[j])'
    end
    if shuffle
        out = out[randperm(n),:]
    end
    return out
end


function AMIS(S, W, k, log_f; epochs=5, nodisp=true)
    IS_tilt = 2.0
    n, p = size(S)
    
    begin
    km = kmeans(copy(S'), k, weights=W)
    
    cmus = zeros(k,2)
    ccovs = zeros(2,2,k)
    for i in range(1, stop=k)
        ixs = findall(x -> isequal(x,i), km.assignments)
        cX = S[ixs, :]; cw = ProbabilityWeights(W[ixs])
        cmus[i,:] = cX' * cw/cw.sum
        ccovs[:,:,i] = cov(cX, cw, corrected=true)
    end
    cpis = [countmap(km.assignments)[i] for i in 1:6]/10
    
    if !nodisp
        f, axs = PyPlot.subplots(5,3, figsize=(8,12))

#         plot_level_curves_all(mus, UTs, ax=axs[1,1], color="red")
        for i = 1:k
            axs[1,1][:plot](splat(gaussian_2D_level_curve(cmus[i,:], ccovs[:,:,i]))...);
        end
    end
    
    ν_S = S; ν_W = W;
    end
    
    nsmp=1000
    
    for i = 1:epochs
        cpis, cmus, ccovs = gmm_custom(ν_S, ν_W, cpis, cmus, ccovs; max_iter=3, tol=1e-3, verbose=false);
        ν_S = sample_from_gmm(1000, cpis, cmus, ccovs*IS_tilt, shuffle=false)
        
        ν_W = log_f(ν_S) - gmm_llh(ν_S, 1, cpis, cmus, ccovs*IS_tilt);
        ν_W = fastexp.(ν_W);
        @noopwhen (nodisp || i > 5) ax = axs[(i ÷ 2) + 1, (i % 2)+1]
        @noopwhen (nodisp || i > 5) plot_is_vs_target(ν_S, ν_W, ax=ax);
        @noopwhen (nodisp || i > 5) for j = 1:6 ax[:plot](splat(gaussian_2D_level_curve(cmus[j,:], ccovs[:,:,j]))...); end
        @noopwhen (nodisp || i > 5) axs[i, 3][:scatter](splat(ν_S)..., alpha=0.2);
    end
    return ν_S, ν_W, cpis, cmus, ccovs
end

function GMM_IS(n, pis, mus, covs, log_f)
    S = sample_from_gmm(n, pis, mus, covs, shuffle=false)
    W = log_f(S) - gmm_llh(S, 1, pis, mus, covs);
    return S, fastexp.(W);
end


# ===================================================================================================================
#    COMBINED SMC SAMPLER
# ===================================================================================================================


@with_kw struct smcs_opt
    resample_every::Int64 = 1 
    sqrtdelta::Float64 =1.
    betas::Array{Float64} = [1.]
    test::Bool = false
    grad_delta::Float64 = 1.0
    burnin::Int64 = 0
    n_init::Int64 = 1000
    prior_std::Float64 = 10.
end

@with_kw struct csmcs_opt
    prior_std::Float64 = 10.
    gris_rsmp_every::Int64 = 1 
    gris_sqrtdelta::Float64 =1.
    ais_betas::Array{Float64} = [1.]
    diagnostics::Bool = false
    gris_grad_delta::Float64 = 1.0
    gris_burnin::Int64 = 0
    gris_epochs::Int64 = 60
    gris_nsmp::Int64 = 50
    amis_kcls::Int64 = 6
    amis_epochs::Int64 = 30
    gmm_smp::Int64 = 5000
    gmm_tilt::Float64 = 2.
end

function combined_smcs(f_log_beta, f_log_target, opts)

	@assert isa(opts, csmcs_opt)
	@unpack_csmcs_opt opts
	gris_opts = smcs_opt(resample_every=gris_rsmp_every, sqrtdelta=gris_sqrtdelta,
		betas=ais_betas, test=diagnostics, grad_delta=gris_grad_delta, burnin=gris_burnin,
		prior_std=prior_std)

    S, W = smcs_grad(gris_epochs, gris_nsmp, f_log_beta, opts=gris_opts)
    S, W, pi, mu, cov = AMIS(S, W, amis_kcls, f_log_target, epochs=amis_epochs, nodisp=!diagnostics);
    S, W = GMM_IS(5000, pi, mu, cov, f_log_target)

    return S, W, [pi, mu, cov]
end

