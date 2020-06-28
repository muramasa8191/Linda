include("linda.jl")
include("dataset/ptb.jl")

using LinearAlgebra

window_size = 2
wordvec_size = 100
corpus, word_to_id, id_to_word = Ptb.load_train_data()
vocab_size = length(word_to_id)
println("counting co-occurrence")
C = Linda.create_co_matrix(corpus, vocab_size, window_size)
println("calculating PPMI")
W = Linda.ppmi(C, verbose=true)

F = svd(W)
u, s, v = F;

idx = [i for i in range(1, length=wordvec_size)]
word_vecs = view(u, :, idx)
querys = ["you", "year", "car", "toyota"]

for query in querys
  Linda.most_similar(query, word_to_id, id_to_word, word_vecs)
end
