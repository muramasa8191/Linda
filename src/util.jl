"""
    preprocess(text)

Apply preprocess to the corpus in order to use natural langage processing.
"""
function preprocess(text::String)
  text = lowercase(text)
  text = replace(text, "." =>  " .")
  words = split(text)

  word_to_id = Dict{String, Int}()
  id_to_word = Dict{Int, String}()

  for word in words
    id = get!(word_to_id, word, word_to_id.count + 1)
    id_to_word[id] = word
  end

  corpus = Vector{Int}()
  for word in words
    append!(corpus, word_to_id[word])
  end

  return corpus, word_to_id, id_to_word
end

"""
    create_co_matrix(corpus, vocab_size, window_size)

Create cosine similarity matrix.
"""
function create_co_matrix(corpus::Vector{Int}, vocab_size::Int, window_size::Int=1)
  (corpus_size,) = size(corpus)
  co_matrix = zeros(Int32, vocab_size, vocab_size)

  for idx in range(1, length = corpus_size)
    word_id = corpus[idx]
    for i in range(1, length = window_size)
      left_idx = idx - i
      right_idx = idx + i

      if left_idx > 0
        left_word_id = corpus[left_idx]
        co_matrix[word_id, left_word_id] += 1
      end
      if right_idx <= corpus_size
        right_word_id = corpus[right_idx]
        co_matrix[word_id, right_word_id] += 1
      end
    end
  end
  return co_matrix
end

using LinearAlgebra.BLAS

"""
    cos_similarity(x, y, eps)

Calculate cosine similaryty.
"""
function cos_similarity(x, y, eps=1e-8)
  nx = x ./ (sqrt(sum(x .^2)) + eps)
  ny = y ./ (sqrt(sum(y .^2)) + eps)

  return BLAS.dot(nx, ny)
end

"""
    most_similar(query, word_to_id, id_to_word, word_matrix, top)

Find k-top similar words to the query given.
"""
function most_similar(query, word_to_id, id_to_word, word_matrix, top::Int=5)
  if get(word_to_id, query, -1) == -1
    println(query, " is not found")
    return
  end
  println("\n[query] ", query)
  query_id = word_to_id[query]
  query_vec = view(word_matrix, query_id, :)

  vocab_size = id_to_word.count
  similaryty = []

  for i in range(1, length = vocab_size)
    append!(similaryty, [(cos_similarity(view(word_matrix, i, :), query_vec), i)])
  end
  similaryty = sort(similaryty, by = x -> x[1], rev = true)
  count = 0
  for (co, id) in similaryty
    if id_to_word[id] == query
      continue
    end
    println(" ", id_to_word[id], ": ", co)
    count += 1
    if (count >= top)
      break
    end
  end
end

"""
    ppmi(co_matrix, verbose, eps)

Calculate PPMI.
"""
function ppmi(co_matrix; verbose = false, eps=1e-8)
  M = zeros(size(co_matrix))
  N = sum(co_matrix)
  S = sum(co_matrix, dims=2)
  total = reduce(*, size(co_matrix), init=1)
  cnt = 0

  for i in range(1, length=size(co_matrix)[1])
    for j in range(1, length=size(co_matrix)[2])
      pmi = log2(co_matrix[i, j] * N / (S[j] * S[i]) + eps)
      M[i, j] = max(0, pmi)

      if verbose && total >= 100
        cnt += 1
        if rem(cnt, (div(total, 100))) == 0
          println((100*cnt/total), "% done")
        end
      end
    end
  end
  return M
end

"""
    create_contexts_target(corpus, window_size)

Create contexts and corresponding targets for the word2vec.
"""
function create_contexts_target(corpus::Vector, window_size::Int = 1)
  target = corpus[window_size+1:end-window_size]
  contexts = zeros(Int, length(corpus) - (2 * window_size), 2*window_size)

  for idx in range(window_size+1, length=length(corpus)-2*window_size)
    idx2 = 0
    for t in range(-window_size, length=2*window_size+1)
      if t == 0
        continue
      end
      idx2 += 1
      contexts[idx-window_size, idx2] = corpus[idx + t]
    end
  end

  return contexts, target
end

"""
    one_hot(x::AbstractVector, true_size)

Generate one hot vector whose length is true_size.
"""
function one_hot(x::AbstractVector,  true_size::Int)
  res = zeros(Int, size(x)[1], true_size)
  for i in range(1, length=size(x)[1])
    res[i, x[i]] = 1
  end
  return res
end

"""
    one_hot(x::AbstractMatrix, true_size)

Generate one hot vector whose length is true_size.
"""
function one_hot(x::AbstractMatrix, true_size::Int)
  res = zeros(Int, size(x)[1], size(x)[2], true_size)
  for i in range(1, length=size(x)[1])
    for j in range(1, length=size(x)[2])
      res[i, j, x[i, j]] = 1
    end
  end
  res
end
