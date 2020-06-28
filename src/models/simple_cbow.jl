using ..Layers
using ..Losses

struct SimpleCBOW
  model::AbstractArray
  loss_layer::Losses.SoftmaxWithLoss
end

function simple_cbow(vocab_size::Int, hidden_size::Int)
  W_in = rand(vocab_size, hidden_size) .* 0.01
  W_out = rand(hidden_size, vocab_size) .* 0.01

  return SimpleCBOW(
    [Layers.matmul(W_in),
    Layers.matmul(W_in),
    Layers.matmul(W_out)],
    Losses.softmax_with_loss())
end

function predict(net::SimpleCBOW, contexts::AbstractArray)
  h0 = net.model[1](view(contexts, :, 1, :))
  h1 = net.model[2](view(contexts, :, 2, :))
  h = broadcast(*, (h0 .+ h1), 0.5)
  score = net.model[3](h)

  return score
end

function (net::SimpleCBOW)(contexts::AbstractArray, target::AbstractArray)
  score = predict(net, contexts)
  loss = net.loss_layer(score, target)

  return loss
end

function backward(net::SimpleCBOW, dout = [1])
  ds = Losses.backward(net.loss_layer, dout)
  da = Layers.backward(net.model[3], ds)
  da = da .* 0.5
  Layers.backward(net.model[1], da)
  Layers.backward(net.model[2], da)
end
