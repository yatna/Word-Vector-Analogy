import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    A = tf.reduce_sum(tf.multiply(inputs,true_w),axis=1)
    temp = tf.matmul(inputs,tf.transpose(true_w))
    B = tf.log(tf.reduce_sum(tf.exp(temp),axis=1))
    
    return tf.subtract(B, A)


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    bs = inputs.get_shape().as_list()[0]
    es = inputs.get_shape().as_list()[1]

    unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    sample = tf.convert_to_tensor(sample, dtype=tf.int32)
    k = sample.get_shape().as_list()[0]
    # # print("----->",bs)
    # sm = [10.0 for i in range(bs)]
    # sm = tf.reshape(tf.convert_to_tensor(sm),[bs,1])
    small_values = [0.000000000001 for j in range(k)]
    sample_additive2 = [0.000000000001 for j in range(bs)]
    small_value3 = [[0.000000000001 for j in range(k)] for i in range(bs)]
    sm = tf.reshape(tf.convert_to_tensor(small_values, dtype=tf.float32), [1,k])
    sm2 = tf.reshape(tf.convert_to_tensor(sample_additive2, dtype=tf.float32), [bs, 1])
    sm3 = tf.reshape(tf.convert_to_tensor(small_value3, dtype=tf.float32), [bs, k])

    # creating additional tensors for further calculation
    one_matrix = [[1.0 for j in range(k)] for i in range(bs)]
    one_tensor = tf.reshape(tf.convert_to_tensor(one_matrix, dtype=tf.float32), [bs,k])

    print("sm ,sm2, sm3",sm.get_shape(),sm2.get_shape(),sm3.get_shape())
    
    u_o = tf.reshape(tf.nn.embedding_lookup(weights,labels),[bs,es]) #[batch_size, embedding_size].
    print("u_o [batch_size,embedding_size]",u_o.get_shape()) 
    b_o = tf.nn.embedding_lookup(biases,labels) #[batch_size, 1].
    print("b_o [batch_size,1]",b_o.get_shape())
    uo_uc = tf.reshape(tf.reduce_sum(tf.multiply(u_o,inputs),axis=1),[bs,1]) # [batch_size,1]
    print("uo_uc [batch_size,1]",uo_uc.get_shape())
    s_a = tf.add(uo_uc,b_o)  #s_a = [batch_size,1]
    print("s_a [batch_size,1]",s_a.get_shape()) #
    p_a = tf.log(tf.add(tf.scalar_mul(k,tf.nn.embedding_lookup(unigram_prob,labels)), sm2)) #p_a =[batch_size,1]
    print("p_a [batch_size,1]",p_a.get_shape())
    A = tf.sigmoid(tf.subtract(s_a , p_a)) #[batch_size,1]
    A = tf.log(tf.add(A,sm2))
    print("A",A.get_shape())
    
    print("==========================")
    u_x = tf.reshape(tf.nn.embedding_lookup(weights,sample),[k,es])#[k, embedding_size].
    print("u_x [k, embedding_size].",u_x.get_shape())
    b_x = tf.reshape(tf.nn.embedding_lookup(biases,sample),[k,1])  #[k,1]
    print("b_x [k, 1].",b_x.get_shape())
    temp = tf.transpose(tf.matmul(u_x,tf.transpose(inputs))) # [k,embed_size]*[embed_size,batch_size]T = [batch_sze,k]
    print("temp [batch_size, k].",temp.get_shape())
    s_x = tf.add(temp, tf.transpose(b_x)) #[batch_size,k] + [1,k] = [batch_size,k]
    print("s_x [batch_size, k]",s_x.get_shape())
    t2 = tf.reshape(tf.nn.embedding_lookup(unigram_prob,sample),[k,1])#[k,1]
    print("t2 [k,1]",t2.get_shape())
    p_x = tf.log(tf.add(tf.scalar_mul(k,tf.transpose(t2)), sm)) #[1,k]
    print("p_x [1,k]",p_x.get_shape())
    b = tf.log(tf.add(tf.subtract(one_tensor, tf.sigmoid(tf.subtract(s_x, p_x))),sm3))  #[batch_size,k]
    print("b [batch_size,k]",b.get_shape())
    B = tf.reshape(tf.reduce_sum(b,axis=1),[bs,1]) #[batch_size,1]
    print("B [batch_size,1]",B.get_shape())
    
    ans = tf.scalar_mul(-1.0,tf.add(A,B)) #[batch_size,1]
    print("Ans [batch_size,1]",ans.get_shape())

    return ans