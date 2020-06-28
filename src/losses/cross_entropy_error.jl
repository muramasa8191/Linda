using LinearAlgebra

"""
    SoftmaxWithLoss(x, t)

struct representing the conbination of the Softmax activation and cross entropy loss.
"""
mutable struct SoftmaxWithLoss
  y::Union{AbstractArray, Nothing}
  t::Union{AbstractArray, Nothing}
end

"""
    softmax_with_loss()

Construct SoftmaxWithLoss object.

# Examples
```julia-repl
julia> softmax_with_loss()
SoftmaxWithLoss(nothing, nothing)
```
"""
function softmax_with_loss()
  SoftmaxWithLoss(nothing, nothing)
end

function (swl::SoftmaxWithLoss)(x::AbstractArray, t::AbstractArray)
  y = softmax(x)
  loss = cross_entropy_error(y, t)
  swl.y = y
  swl.t = t
  return loss
end

"""
    backward(swl, dout)

Backward propagation for Cross entropy loss and softmax.
"""
function backward(swl::SoftmaxWithLoss, dout::AbstractArray)
  batch_size = size(swl.t)[1]
  dx = swl.y .- swl.t
  broadcast(/, dx, batch_size)
end

"""
    cross_entropy_error(y::AbstractVector, t::AbstractVector)

Calculate Cross Entropy Error for Vector.
# Examples
```julia-repl
julia> x = [0.5713270306991245, 0.02281998489356884, 0.10240377749202871, 0.24070864883516674, 0.06274055808011111]
5-element Array{Float64,1}:
 0.5713270306991245
 0.02281998489356884
 0.10240377749202871
 0.24070864883516674
 0.06274055808011111

julia> cross_entropy_error(x, [0, 0, 1, 0, 0])
3.2876577513012926
```
"""
function cross_entropy_error(y::AbstractVector, t::AbstractVector) 
  ε = 1e-7
  -dot(map(log2, y .+ ε), t)
end

"""
    cross_entropy_error(y, t)

Calculate Cross Entropy Error for Matrix.

# Examples
```julia-repl
julia> x = [0.465746 0.235804 0.29845; 0.0615087 0.127529 0.810962]
2×3 Array{Float64,2}:
 0.465746   0.235804  0.29845
 0.0615087  0.127529  0.810962

julia> cross_entropy_error(x, [0 0 1; 1 0 0])
2.883750858691105
```
"""
function cross_entropy_error(y::AbstractMatrix, t::AbstractMatrix)
  ε = 1e-7
  batch_size = size(y, 1)
  -dot(map(log2∘(x)->x+ε, y), t) / batch_size
end

"""
    softmax(x::AbstractVector)

Softmax function for Vector.

# Examples
```julia-repl
julia> x = [-0.31676190821060674 0.2519547815711753 -0.6186712932474587 -0.9380339825044721 0.7497297487757365]
1×5 Array{Float64,2}:
 -0.316762  0.251955  -0.618671  -0.938034  0.74973

julia> softmax(x)
1×5 Array{Float64,2}:
 0.14393  0.25418  0.106422  0.0773278  0.41814
```
"""
function softmax(x::AbstractVector)
  c = maximum(x)
  x = map(exp, x .- c)
  return x ./ sum(x)
end

"""
    softmax(x::AbstractMatrix)

Softmax function for Matrix.
# Examples
```julia-repl
julia> x = [-0.524002 0.7206 -1.51316; -2.02896   -0.902642  -0.128476]
2×3 Array{Float64,2}:
 -0.524002   0.7206    -1.51316
 -2.02896   -0.902642  -0.128476

julia> softmax(x)
2×3 Array{Float64,2}:
 0.206465   0.716753  0.0767821
 0.0928211  0.286286  0.620893
"""
function softmax(x::AbstractMatrix)
  mapslices(softmax, x, dims=[2])
end