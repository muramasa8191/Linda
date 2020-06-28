"""
    SGD(lr)

A structure representing stochastic gradient descent(SGD).
"""
struct SGD
  lr::AbstractFloat
end

function (sgd::SGD)(layer)
  for i in range(1, length=length(layer.params))
    layer.params[i] = layer.params[i] - sgd.lr * layer.grads[i]
  end
end
