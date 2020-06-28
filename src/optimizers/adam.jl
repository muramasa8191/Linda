"""
    Adam(lr, beta1, beta2, iter, m, v)

A structure representing Adam optimizer.
"""
mutable struct Adam
  lr::AbstractFloat
  beta1
  beta2
  iter::Int
  m::AbstractArray
  v::AbstractArray
end

"""
    adam()

Construct Adam optimizer.

# Examples
```julia-repl
julia> adam(0.01)
Adam(0.01, 0.9, 0.999, 0, Any[], Any[])
```
"""
function adam(lr::AbstractFloat=0.001, beta1::AbstractFloat=0.9, beta2::AbstractFloat=0.999)
  Adam(lr, beta1, beta2, 0, [], [])
end

function (adam::Adam)(layer)
  if length(adam.m) == 0
    adam.m = [zeros(size(param)) for param in layer.params]
    adam.v = [zeros(size(param)) for param in layer.params]
  end
  adam.iter += 1
  lr_t = adam.lr * sqrt(1.0 - adam.beta2^adam.iter) / (1.0 - adam.beta1^adam.iter)

  for i in range(1, length=size(layer.params)[1])
    for j in range(1, length=length(layer.params[i]))
        adam.m[i][j] += (1 - adam.beta1) .* (layer.grads[i][j] - adam.m[i][j])
        adam.v[i][j] += (1 - adam.beta2) .* (layer.grads[i][j]^2 - adam.v[i][j])
        layer.params[i][j] -= lr_t * adam.m[i][j] / (sqrt(adam.v[i][j]) + 1e-7)
    end
  end
end
