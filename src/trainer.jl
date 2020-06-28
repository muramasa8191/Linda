using Dates
using Random
using .Models

"""
    Trainer(model, optimizers)

A structure to train model given by using the optimizers.
The number of optimizers should be same as the number of layers in the model given.
"""
struct Trainer
 model
 optimizers
end

"""
    get_batch(x, idx, batch_size, i)

Retrieve the i-th batch whose indices in the dataset are idx and the size is batch_size.

# Examples
```julia-repl
julia> get_batch([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], [3, 1, 2], 2, 1)
2×3 view(::Array{Float64,2}, [3, 1], :) with eltype Float64:
 7.0  8.0  9.0
 1.0  2.0  3.0
```
"""
function get_batch(x::AbstractArray{T, 2}, idx::AbstractArray, batch_size::Int, i::Int) where T <: Number
  view(x, idx[batch_size * (i - 1)+1:min(batch_size * i, end)], :)
end

"""
    get_batch(x, idx, batch_size, i)

Retrieve the i-th batch whose indices in the dataset are idx and the size is batch_size.

# Examples
```julia-repl
julia> get_batch(reshape([i * 1.0 for i in range(1, length=8)], 2, 2, 2), [2, 1], 1, 2)
1×2×2 view(::Array{Float64,3}, [1], :, :) with eltype Float64:
[:, :, 1] =
 1.0  3.0

[:, :, 2] =
 5.0  7.0
```
"""
function get_batch(x::AbstractArray{T, 3}, idx::AbstractArray, batch_size::Int, i::Int) where T <: Number
  view(x, idx[batch_size * (i - 1)+1:min(batch_size * i, end)], :, :)
end

function (trainer::Trainer)(x::AbstractArray, t::AbstractArray, epochs::Int,
    batch_size::Int = 32; max_grads::Union{AbstractFloat, Nothing}=nothing, eval_interval::Int=10)
  data_size = size(x)[1]
  max_iter = div(data_size, batch_size)

  model = trainer.model
  optimizers = trainer.optimizers

  total_loss = 0.0
  start_time = now()
  for epoch in range(1, length=epochs)
    idx = randperm(MersenneTwister(1234), data_size)
    for i in range(1, length=max_iter)
      batch_x = get_batch(x, idx, batch_size, i)
      batch_t = view(t, idx[batch_size * (i - 1)+1:min(batch_size * i, end)], :)

      loss = model(batch_x, batch_t)  
      total_loss += loss

      grads = Linda.Models.backward(model)

      for j in range(1, length=length(model.model))
        optimizers[j](model.model[j])
      end
      if (i % eval_interval == 0)
        avg_loss = total_loss / max_iter
        total_loss = 0.0
        println("| epoch ", epoch, " |  iter ", i , "/", max_iter, " | time ", now() - start_time, " | loss ", avg_loss)
      end
    end
  end
  return model
end