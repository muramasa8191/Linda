"""
    Affine(params, grads, nothing)

Linear Layer that represents y = WX + B.

# Examples
```julia-repl
julia> Affine(zeros(2, 3), zeros(1, 3), nothing)
Affine([0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0 0.0], nothing)
```
"""
mutable struct Affine
  params::AbstractArray
  grads::AbstractArray
  x::Union{AbstractArray, Nothing}
end

"""
    affine(w, b)

Contruct an Affine Layer.

# Examples
```julia-repl
julia> affine(zeros(3, 4), zeros(1, 4))
Affine([[0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0], [0.0 0.0 0.0 0.0]], [[0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0], [0.0 0.0 0.0 0.0]], nothing)
```
"""
function affine(w::AbstractArray, b::AbstractArray)
  return Affine([w, b], [zeros(size(w)), zeros(size(b))], nothing)
end

function (affine::Affine)(x::AbstractArray)
  W, b = affine.params
  affine.x = x
  return x * W .+ b
end

"""
    backward(affine, dout)

Backward propagation of Affine Layer.
"""
function backward(affine::Affine, dout::AbstractArray)
  dx = dout * affine.params[1]'
  dW = affine.x' * dout
  db = sum(dout, dims=1)
  affine.grads[1] = dW
  affine.grads[2] = db
  return dx
end
