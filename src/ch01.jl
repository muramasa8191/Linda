using Random

include("linda.jl")
include("dataset/spiral.jl")

model = Linda.Models.two_layer_net(2, 10, 3)
x, t = Spiral.load_data()

max_epoch = 300
batch_size = 30

optimizers = [Linda.Optimizers.adam(0.5) for i in range(1, length=length(model.model))]
max_iter = div(size(x)[1], batch_size)

for epoch in range(1, length=max_epoch)
  idx = randperm(MersenneTwister(1234), size(x)[1])
  total_loss = 0.0
  for i in range(1, length=max_iter)
    batch_x = view(x, idx[batch_size * (i - 1)+1:batch_size * i], :)
    batch_t = view(t, idx[batch_size * (i - 1)+1:batch_size * i], :)

    loss = model(batch_x, batch_t)  
    total_loss += loss

    grads = Linda.Models.backward(model, [1])

    layers = []
    for j in range(1, length=3)
      optimizers[j](model.model[j])
    end

    if (i % 10 == 0)
      avg_loss = total_loss / 10
      println("| epoch ", epoch, " |  iter ", i , "/", max_iter, " | loss ", avg_loss)
    end
  end
end