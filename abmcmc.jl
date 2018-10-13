module abmcmc

using LinearAlgebra
using Flux
using Flux: Tracker, params
using Flux.Tracker: @grad
using DSP: conv
using StatsBase: autocor

function mala(logdensity, h, M, iters, init_proposal_mean; δ_trunc=Inf)
    
    function log_llh_and_proposal(z)
        par_z = param(z)
        logp_u = logdensity(par_z)
        Tracker.back!(logp_u)
        grad = Tracker.grad(par_z)
        
        #  ----- Truncated MALA (T-MALA) ----
        #  ----- (aka norm clipping!) -------
        if δ_trunc < Inf
            grad_norm = sum(x -> x^2, grad)
            if grad_norm > δ_trunc
                grad *= δ_trunc / grad_norm
            end
        end
        return Tracker.data(logp_u), z + 0.5*h*M*grad, grad   # prev (unnorm) logp, gradient step, gradient
    end
    
    # set up
    d = length(init_proposal_mean)
    ztrace = Array{Float64, 2}(undef, d, iters)
    cholΣ = Matrix(cholesky(h*M).L)   # `Matrix` keeps Flux happy for the time being. For d < 20, it really makes no difference.
    
    function gauss_logpdf(z)
        u = cholΣ \ z
        exponent = -0.5*sum(x->x*x, u)
#         lognormconst = -0.5*logdet(2π * h * M)
        out = exponent #+ lognormconst
        return out
    end
    gauss_logpdf(z, μ) = gauss_logpdf(z - μ)
    
    # initialise
    accepted = 0
    z = init_proposal_mean
    cur_u_logp = -Inf
    proposal_mean = init_proposal_mean
    prev_proposal_mean = init_proposal_mean
    
    # Markov Chain loop
    for i = 1:iters
        # transition r.v.
        ϵ_prop = randn(d)
        prop_transition_logp = -0.5*sum(x->x*x, ϵ_prop)
        
        # deterministic proposal
        z_prop = proposal_mean + cholΣ * ϵ_prop
        
        # calculate logdensity(z_prop) and accompanying new proposal mean
        prop_u_logp, ν_proposal_mean, _ = log_llh_and_proposal(z_prop)
        
        # metropolis-hastings
        r = prop_u_logp - cur_u_logp + 
                gauss_logpdf(z, ν_proposal_mean) - 
                prop_transition_logp   |> exp
        
        # if <accept>
        if rand() < r
            z = z_prop                                                       # update z
            prev_proposal_mean = proposal_mean                               # store prev proposal mean
            cur_u_logp, proposal_mean = prop_u_logp, ν_proposal_mean
            accepted += 1
        end
        
        ztrace[:,i] = z      # store z regardless
    end
    
    return ztrace, accepted/iters
end



 #=================================================================================
                                DIAGNOSTIC TOOLS
 ==================================================================================#

# not the most efficient AR calculation, but will do until optimisation reqd
ar_coeffs(X, max_l) = reduce(hcat, [X[i:end-max_l+i-1] for i in 1:max_l]) \ X[max_l+1:end]


function spectrum0_ar(x::Array{T,1}; ncoeffs=nothing) where T <: Real
    nrows = length(x)
    ncoeffs = something(ncoeffs, Int64(ceil(10 * log10(nrows)))) # for AR (not doing AIC)
    
    # if column is (almost) perfect line, spectrum is 0
    # (avoids pathologies in AR calculation)
    if isapprox(std(diff(x)), 0.)
        spec = 0.
        ncoeffs = 0
    else
        cx = x .- mean(x)
        coeffs = ar_coeffs(cx, ncoeffs)
        x̂ = vcat(cx[1], conv(cx, coeffs)[1:end-ncoeffs])
        res_var = var(cx) - var(x̂)
        spec = res_var/(1 - sum(coeffs))^2
    end

    return convert(T, spec), ncoeffs
end

#= In order for this to function more effectively, the spectrum0_ar
   function needs to choose the order of AR process via AIC. While not
   especially difficult, it is also not trivial to add this in, esp. in
   an efficient manner, so I have left this as WIP for now.
=#
function effective_size_ar(x::Array{T,1}) where T <: Real
    @warn "Currently WIP. Need to implement AIC for AR model selection."
    spec, _order = spectrum0_ar(x)
    m = length(x)
    ess = spec != 0. ? m * var(x) / spec : 0.
    return ess
end


function effective_size_asymp(x::Array{T,1}) where T <: Real
    # see e.g. BDA3 (sec 11.5) or Thompson 2010: A comparison of ... autocorrelation time.
    n = length(x)
    maxlag = min(n, Int64(ceil(30*log10(length(x)))))  # heuristic max
    ρ = autocor(x, 1:maxlag)
    
    # Geyer, 1992: sums of consecutive ACF values are > 0.
    consec_pairs = conv(ρ, ones(2))
    ncoeffs = findfirst(consec_pairs .< 0) - 1
    return  n / (1 + 2*(sum(ρ[1:ncoeffs])))
end


# coda's version of Rhat is really nice. Converted approx. as-is.
function Rhat(xs...; confidence = 0.95) 
    #=================================================================
    Unlike BDA3, Plummer et al. in R's coda package use an additional
    adjustment for 1/nchains, as well as add some CIs.
    =================================================================#
    @assert (length(unique(map(length, xs))) == 1) "inconsistent lengths in xs"
    x = reduce(hcat, xs)
    n, nchains = size(x)
    
    #=================================================================
    Estimate mean within-chain variance (W) and between-chain variance
    (B/Niter), and calculate sampling variances and covariance of the
    estimates (varW, varB, covWB)
    =================================================================#
    ## Univariate (lower case)
    s2 = var(x, dims=1)
    w  = mean(s2)
    x̄  = mean(x, dims=1)
    b  = n * var(x̄)

    muhat  = mean(x̄)
    var_w  = var(s2)/nchains              
    var_b  = (2 * b^2)/(nchains - 1)      
    cov_wb = (n/nchains) * cov(s2[:], x̄[:].^2) - 2 * muhat * cov(s2[:], x̄[:])

    #=================================================================  
    Posterior interval combines all uncertainties in a t interval with
    center muhat, scale sqrt(V), and df_V degrees of freedom.
    =================================================================#
    V     = (n-1)w / n  + (1 + 1/nchains)b / n
    var_V = (var_w * (n-1)^2 + 
           (1 + 1/nchains)^2 * var_b + 2 * (n-1) * (1 + 1/nchains) * cov_wb
          )/n^2
    df_V = 2V^2 / var_V

    #=================================================================
                      R = sqrt(V/W) * df_adj
    where df_adj is a degrees of freedom adjustment for the width
    of the t-interval.

    To calculate upper confidence interval we divide R2 = R^2 into two
    parts, fixed and random.  The upper limit of the random part is
    calculated assuming that B/W has an F distribution.
    =================================================================#

    df_adj      = (df_V + 3)/(df_V + 1)
    B_df        = nchains - 1
    W_df        = (2 * w^2)/var_w
    R2_fixed    = (n - 1)/n
    R2_random   = (1 + 1/nchains) * (1/n) * (b/w)
    R2_estimate = R2_fixed + R2_random
    R2_upper    = R2_fixed + quantile(FDist(B_df, W_df), (1 + confidence)/2) * R2_random
    R̂           = sqrt(df_adj * R2_estimate)
    R̂_ci_upper  = sqrt(df_adj * R2_upper)

    return R̂, R̂_ci_upper
end

end