import os
import pickle
import numpy as np
from scipy import spatial


model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
f = open("word_analogy_test.txt",'r')
ans = open("pair_pred.txt",'w+')
x = f.readlines()
embedding_size = embeddings.shape[1]
print(embedding_size)
count = 0
for i in x:
    a, b = i.split('||')
    examples = [i[1:-1] for i in a.split(',')]
    examples = [tuple(i.split(':')) for i in examples]
    total_diff = np.zeros(embedding_size,dtype=float)
    # print(examples)
    for k,v in examples:
        # print(k, embeddings[dictionary[k]].shape)
        embed_k = embeddings[dictionary[k]]
        embed_v = embeddings[dictionary[v]]
        diff = embed_k - embed_v
        total_diff += diff/sum(diff**2)**0.5
        # total_diff +=diff

    options = [i.rstrip()[1:-1] for i in b.split(',')]
    options = [tuple(i.split(':')) for i in options]
    ans_min = 100000000000000
    ans_max = -10000000000000
    # print(options)
    min_tuple = max_tuple = options[0]
    for k, v in options:
        embed_k = embeddings[dictionary[k]]
        embed_v = embeddings[dictionary[v]]
        single_diff = embed_k - embed_v
        single_diff = single_diff/sum(single_diff**2)**0.5
        # correlation = 1 - spatial.distance.cosine(diff,single_diff)
        correlation = np.sum(single_diff * diff)
        if correlation > ans_max:
            max_tuple = (k, v)
            ans_max = correlation
        if correlation < ans_min:
            min_tuple = (k, v)
            ans_min = correlation
    options = options + [min_tuple, max_tuple]
    s = " ".join(['"' + k + ":" + v + '"' for (k, v) in options])
    ans.write(s + '\n')
    # print("Count", count, s)
    # count +=1
    # if count==5:
    #     break

f.close()
ans.close()



