"""
    softmax(x)

Softmax Activation for Vector.
"""
function softmax(x::AbstractVector)
  c = maximum(x)
  x = map(exp, x .- c)
  return x ./ sum(x)
end

"""
    softmax(x)

Softmax Activation for Matrix
"""
function softmax(x::AbstractMatrix)
  mapslices(softmax, x, dims=[2])
end
