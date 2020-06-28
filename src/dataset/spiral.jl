"""
Module that generates Spiral data.
"""
module Spiral

using Random

"""
    load_data(seed)

Generate the data in spiral.
"""
function load_data(seed=1984)
  Random.seed!(seed)
  N = 100  # Samples for each class
  DIM = 2  # dimension of the data
  CLS_NUM = 3  # classes

  x = zeros(N*CLS_NUM, DIM)
  t = zeros(Int, N*CLS_NUM, CLS_NUM)

  for j in range(1, length=CLS_NUM)
    for i in range(1, length=N) #N*j, N*(j+1)):
      rate = (i-1) / N
      radius = 1.0*rate
      theta = (j-1)*4.0 + 4.0*rate + rand()*0.2

      ix = N*(j-1) + i
      x[ix, 1:DIM] = [radius*sin(theta), radius*cos(theta)]
      t[ix, j] = 1
    end
  end
  return x, t
end

end
