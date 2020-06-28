using ..Layers
using ..Losses


"""
    TwoLayerNet(model, loss_layer)

A structure consisting of two affine layers.
"""
struct TwoLayerNet
  model::AbstractArray
  loss_layer
end

"""
    two_layer_net(input_size, hidden_size, output_size)

Construct TwoLayerNet with given number of parameters.
"""
function two_layer_net(input_size::Int, hidden_size::Int, output_size::Int)
  W1 = rand(input_size, hidden_size) .* 0.01
  b1 = zeros(1, hidden_size)
  W2 = rand(hidden_size, output_size) .* 0.01
  b2 = zeros(1, output_size)

  return TwoLayerNet(
    [Layers.affine(W1, b1), Layers.sigmoid(), Layers.affine(W2, b2)],
     Losses.softmax_with_loss())
end

"""
    predict(net, x)

Predict the consequence of input x.
"""
function predict(net::TwoLayerNet, x::AbstractArray)
  for layer in net.model
    x = layer(x)
  end
  return x
end

function (net::TwoLayerNet)(x::AbstractArray, t::AbstractArray)
  x = predict(net, x)
  loss = net.loss_layer(x, t)

  return loss
end

"""
    backward(net, dout) 

Apply backward propagation for each layers in the model.
"""
function backward(net::TwoLayerNet, dout = [1])
  dout = Losses.backward(net.loss_layer, dout)
  for layer in reverse(net.model)
    dout = Layers.backward(layer, dout)
  end
end
