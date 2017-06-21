import dynet as dy
import numpy as np

# Initialize a model for scnn
nmf = 40
nframes = 1024
nlabel = 61
J = [nmf, 64, 512, 1024, nlabel]
N = [4, 24, 24]
D = [nframes, nframes/2, nframes/4, nframes/4]
batch_size = 128

dy.renew_cg()
scnn = dy.Model()
DO_BATCH = False
PRINT_EMBED = True
PRINT_SIM = False
DEBUG = True

# 1) Add in some weights and biases
pw_i = scnn.add_parameters((1, N[0]+1, J[0], J[1]))
pb_i = scnn.add_parameters((J[1]))

pw_h1 = scnn.add_parameters((1, N[1]+1, J[1], J[2]))
pb_h1 = scnn.add_parameters((J[2]))

pw_h2 = scnn.add_parameters((1, N[2]+1, J[2], J[3]))
pb_h2 = scnn.add_parameters((J[3]))

pw_o = scnn.add_parameters((J[3], J[4]))
pb_o = scnn.add_parameters((J[4]))

# 2) Form the expressions for the parameters
w_i = dy.parameter(pw_i)
b_i = dy.parameter(pb_i)

w_h1 = dy.parameter(pw_h1)
b_h1 = dy.parameter(pb_h1)

w_h2 = dy.parameter(pw_h2)
b_h2 = dy.parameter(pb_h2)

w_o = dy.parameter(pw_o)
b_o = dy.parameter(pb_o)

# Initialize a model for vgg
nvgg = 4096
nembed = 1024

pw_embed = scnn.add_parameters((nembed, nvgg))
pb_embed = scnn.add_parameters((nembed))

w_embed = dy.parameter(pw_embed)
b_embed = dy.parameter(pb_embed)

# Train the network
# 0) Helper functions for minibatch training and compute accuracy from similarity matrix
def do_one_batch(X_batch, Z_batch):
    # Flatten the batch into 1-D vector for workaround
    batch_size = X_batch.shape[0]
    if DO_BATCH:
        X_batch_f = X_batch.flatten('F')
        Z_batch_f = Z_batch.flatten('F')
        x = dy.reshape(dy.inputVector(X_batch_f), (nmf, nframes), batch_size=batch_size)
        z = dy.reshape(dy.inputVector(Z_batch_f), (nvgg), batch_size=batch_size)
        scnn.add_input([X_batch[i] for i in range(X_batch.shape[0])])
        vgg.add_input([Z_batch[i] for i in range(X_batch.shape[0])])

    else:
        x = dy.matInput(X_batch.shape[0], X_batch.shape[1])
        x.set(X_batch.flatten('F'))
        z = dy.vecInput(Z_batch.shape[0])
        z.set(Z_batch.flatten('F'))
        x = dy.reshape(dy.transpose(x, [1, 0]), (1, X_batch.shape[1], X_batch.shape[0]))
    print(x.npvalue().shape)
    a_h1 = dy.conv2d_bias(x, w_i, b_i, [1, 1], is_valid=False)
    h1 = dy.rectify(a_h1)
    h1_pool = dy.kmax_pooling(h1, D[1], d=1)

    a_h2 = dy.conv2d_bias(h1_pool, w_h1, b_h1, [1, 1], is_valid=False)
    h2 = dy.rectify(a_h2)
    h2_pool = dy.kmax_pooling(h2, D[2], d=1)

    a_h3 = dy.conv2d_bias(h2_pool, w_h2, b_h2, [1, 1], is_valid=False)
    h3 = dy.rectify(a_h3)
    h3_pool = dy.kmax_pooling(h3, D[3], d=1)

    h4 = dy.kmax_pooling(h3_pool, 1, d=1)
    h4_re = dy.reshape(h4, (J[3],))
    #print(h4_re.npvalue().shape)
    g = dy.scalarInput(1.)
    zem_sp = dy.weight_norm(h4_re, g)
    #print(zem_sp.npvalue().shape)
    zem_vgg = w_embed*z + b_embed
    #print(zem_vgg.npvalue().shape)

    sa = dy.transpose(zem_sp)*zem_vgg
    s = dy.rectify(sa)

    if PRINT_EMBED:
        print('Vgg embedding vector:', zem_vgg.npvalue().shape)
        print(zem_vgg.value())
    
        print('Speech embedding vector:', zem_sp.npvalue().shape)
        print(zem_sp.value())
    if PRINT_SIM:
        print('Raw Similarity:', sa.npvalue())
        print(sa.value())
        print('Similarity:', s.npvalue())
        print(s.value())

    return s
'''def sim2accuracy(s, top=10):
    batch_size = s.shape[0]
    top_indices = zeros(ntop, batch_size)
    for k in range(ntop):
        cur_top_idx = np.argmax(s, axis=1)
        top_indices[k] = cur_top_idx
        # To leave out the top values that have been determined and the find the top values for the rest of the indices
        for l in range(batch_size):
            s[l][cur_top_idx[l]] = -1
            #similarity[cur_top_idx] = -1;
            # Find if the image with the matching index has the highest similarity score
            dev = abs(top_indices - np.linspace(0, 199, 200))
            min_dev = np.amin(dev, axis=0)
            #print('current deviation from correct indices for dev test:', min_dev)
    accuracy = np.mean((min_dev==0))
    return accuracy'''


# 1) Input the dataset with minibatches
captions = np.load('captions_40k.npz')
captions_tr = captions['arr_0'][0:35000]
captions_tx = captions['arr_0'][35000:35200]
images = np.load('images_40k.npz')
images_tr = images['arr_0'][0:35000]
images_tx = images['arr_0'][35000:35200]

batchsize = 128
ntr = 50
nbatch = int(ntr/batchsize)
if DEBUG:
    print(captions_tr[0].shape)
do_one_batch(captions_tr[0], images_tr[0])

'''do_one_batch(captions_tr[0:128], images_tr[0:128])
for i in range(ntr):
    #randidx = np.argsort(np.random.normal(size=(ntr,)), 0)
    for j in range(nbatch):
        cur_idx = randidx[j*nbatch:(j+1)*nbatch]
        X_batch = list(captions_tr[cur_idx])
        Z_batch = list(images_tr[cur_idx])
        sa, s, cost = do_one_batch(X_batch, Z_batch)

        if j % 10 == 0:
            print('Train Accuracy is:', accuracy)
            sa, s, cost = do_one_batch(captions_tx, images_tx)
            accuracy = sim2accuracy(sa.value())
            print('Development Test Accuracy is:', accuracy)'''

niter = 50
# Compute the cost function without minibatch
trainer = dy.SimpleSGDTrainer(scnn)
for i in xrange(niter):
    for j in xrange(nbatch):
        batch_sp = captions_tr[j*batch_size:(j+1)*batch_size]
        batch_im = images_tr[j*batch_size:(j+1)*batch_size]
        similarity = []
        for k in xrange(batch_size):
            similarity_row = []
            wrong_idx = []
            for l in xrange(batch_size):
                cur_sp = batch_sp[k]
                cur_im = batch_im[l]
                cur_s = do_one_batch(cur_sp, cur_im)
                similarity_row.append(cur_s)
                if not k==l:
                    wrong_idx.append(l)
            #idx_c = wrong_idx[]
            #idx_i = wrong_idx[]
            similarity.append(similarity_row)

        for k in xrange(batch_size):
            if k == 0:
                cost = similarity[0][0] - similarity[0][0] + 1
            for l in xrange(batch_size):
                cost = cost + dy.rectify(similarity[k][l] - similarity[k][k] + 1)

            if DEBUG:
                print(cost.value())

        cost.backward()
        trainer.update()
# Save the model
