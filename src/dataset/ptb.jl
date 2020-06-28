"""
Module that handles Pen Tree Bank data.
"""
module Ptb

"""
    load_train_data()

Load Pen Tree Bank train data.
"""
function load_train_data()
  filename = "ptb.train.txt"
  if (!ispath(filename))
    download("https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt", filename)
  end
  text = read(filename, String)
  text = String(strip(replace(text, "\n" => "<eos>")))
  words = split(text)

  word_to_id = Dict{String, Int}()
  id_to_word = Dict{Int, String}()
  corpus = Vector{Int}()

  for word in words
    id = get!(word_to_id, word, word_to_id.count + 1)
    id_to_word[id] = word
    append!(corpus, word_to_id[word])
  end

  rm(filename)

  return corpus, word_to_id, id_to_word
end

"""
    load_test_data()

Load Pen Tree Bank test data.
"""
function load_test_data()
  filename = "ptb.test.txt"
  if (!inpath(filename))
    download("https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt", filename)
  end
  text = read(filename, String)
  return String(strip(replace(text, "\n" => "<eos>")))
end

"""
    load_validation_data()

Load Pen Tree Bank validation data.
"""
function load_validation_data()
  filename = "ptb.valid.txt"
  if (!ispath(filename))
    download("https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt", filename)
  end
  text = read(filename, String)
  return String(strip(replace(text, "\n" => "<eos>")))
end

end