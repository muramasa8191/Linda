"""
    Matmul(params, nothing)

A struct representing Matrix Multiplication.
"""
mutable struct Matmul
  params::AbstractArray
  grads::AbstractArray
  x::Union{AbstractArray, Nothing}
end

"""
    matmul(weight)

Construct Matmul Layer.

# Examples
```julia-repl
julia> matmul(zeros(2,4))
Matmul([[0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]], nothing)
```
"""
function matmul(w::AbstractArray)
  return Matmul([w], [zeros(size(w))], nothing)
end

function (m::Matmul)(x::AbstractArray)
  m.x = x
  return x * m.params[1]
end

function backward(m::Matmul, dout::AbstractArray)
  dx = dout * m.params[1]'
  dW = m.x' * dout
  m.grads[1] = dW
  return dx
end
