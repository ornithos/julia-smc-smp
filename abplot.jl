module abplot

using PyPlot
using Flux: Tracker
include("./abutils.jl")

function _gen_cube_grid(vmin, vmax; nsmp_dim=8, ndim=2, force=false)
    """
    Generate coordinates for equally spaced meshgrid in `ndim` dimensions.
    The output is in 'long' rather than 'wide' form: i.e. if `ndim=2`,
    an n x 2 matrix is returned.
    """
    (!force) && nsmp_dim^ndim > 1e4 && throw("More than 10,000 points requested.")
    xs = collect(range(vmin, stop=vmax, length=nsmp_dim))
    xs = hcat(repeat(xs, outer=(nsmp_dim,1)), repeat(xs, inner=(nsmp_dim,1)))
    return xs
end


function _gen_grid(xrng, yrng; nsmp_dim=8, ndim=2, force=false)
    """
    Generate coordinates for equally spaced meshgrid in `ndim` dimensions.
    The output is in 'long' rather than 'wide' form: i.e. if `ndim=2`,
    an n x 2 matrix is returned.
    """
    (!force) && nsmp_dim^ndim > 1e4 && throw("More than 10,000 points requested.")
    xs = collect(range(xrng[1], stop=xrng[2], length=nsmp_dim))
    ys = collect(range(yrng[1], stop=yrng[2], length=nsmp_dim))
    return hcat(repeat(xs, outer=(nsmp_dim,1)), repeat(ys, inner=(nsmp_dim,1)))
end

function _tile_image_grid(Ms; gridsz=nothing)
    
    num = length(Ms)
    # primarily to catch user fails of passing in a single array
    @assert num <= 256 "too many images. Expecting at most 256."
    @assert ndims(Ms[1]) == 2 "each element of Ms should be a 2 dimensional image array"
    @assert maximum(size(Ms[1]))*sqrt(float(num)) <= 4000 "resulting image array is too large. Reduce number of Ms."
    @assert length(unique([size(Ms[i]) for i in 1:num])) == 1 "all arrays in Ms should be same size."
    
    Ms_sz = size(Ms[1])
    
    if gridsz == nothing
        poss = [[x,Int(ceil(num/x))] for x in range(1,stop=Int(floor(sqrt(num)))+1)]
        resultsz = [x .* Ms_sz for x in poss]
        choice = findmin([sum(x) for x in resultsz])[2]
        gridsz = sort(poss[choice])
    else
        gridsz = collect(gridsz)
    end
    @assert isa(gridsz, Array{Int, 1}) && size(gridsz) == (2,)
    
    gridsz = (gridsz == nothing) ? abutils.subplot_gridsize(num) : gridsz

    
    out = zeros((gridsz .* Ms_sz)...)
    for ii in 1:gridsz[1], jj in 1:gridsz[2]
        i = (ii-1) * gridsz[2] + jj
        ix_xs = ((gridsz[1] -ii )*Ms_sz[1] + 1) : ((gridsz[1] - ii +1)*Ms_sz[1])
        ix_ys = ((jj-1)*Ms_sz[2] + 1) : (jj*Ms_sz[2])
        out[ix_xs, ix_ys] = Ms[i]   
    end
    return copy(out)  # not sure if the copy is necessary. Freq is for PyPlot.
end

arange(start; kwargs...) = collect(range(start; kwargs...))
function plot_2dtile_imshow(forward_network; nsmp_dim=8, scale=2., xrng=nothing, yrng=nothing,
        im_shp=(28,28), transpose=false, cmap="viridis", ax=nothing)
    """
    :param forward_network: decoder for VAE. Should accept 2 x n matrices.
    :param nsmp_dim: number of (equally spaced) samples in both x and y dims.
    :param scale: scale of grid on latent space (will be [-scale, scale] in both dims)
    :param im_shp: size/shape of each image. Default = (28,28) for MNIST.
    :param transpose: images need transposing before displaying (e.g. if in row major format)
    :param cmap: colormap -- default "viridis", suggest also "binary_r" for greyscale.
    """
    @assert (xrng == nothing && yrng == nothing) || (xrng != nothing && yrng != nothing) "If " *
        "either xrng or yrng specified, both must be specified."
    xrng != nothing && scale != 2. && @warn "Ignoring scale specification, as rng given."
    xrng = xrng == nothing ? [-scale, scale] : xrng
    yrng = yrng == nothing ? [-scale, scale] : yrng
    
    zs = _gen_grid(xrng, yrng, nsmp_dim=nsmp_dim)
    tr = (transpose == true ? Base.transpose : x -> x)
    ims = [reshape(Tracker.data(forward_network(zs[ii,:])), im_shp) for ii in 1:size(zs,1)]
    imtiled = tr(_tile_image_grid(ims))
    ax = ax == nothing ? gca() : ax
    ax[:imshow](imtiled, cmap=cmap)
    Lx = max(all([typeof(x) <: Int for x in xrng]) ? diff(xrng)[1]+1 : 3, 3)
    Ly = max(all([typeof(x) <: Int for x in yrng]) ? diff(yrng)[1]+1 : 3, 3)
    ax[:set_xticks](arange(1,stop=im_shp[1]*nsmp_dim, length=Lx));  ax[:set_xticklabels](arange(xrng[1],stop=xrng[2], length=Lx))
    ax[:set_yticks](arange(1,stop=im_shp[2]*nsmp_dim, length=Ly));  ax[:set_yticklabels](arange(yrng[2],stop=yrng[1], length=Ly))
    return ax
end

end