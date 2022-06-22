import os, pickle, sys
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("bright", 2)


def main(direc):

    dirout = direc.replace('embeddings','umap_tsne')
    if not os.path.exists(dirout): os.makedirs(dirout)

    if os.path.exists(direc+'/emb.pkl'):
        os.remove(direc+'/emb.pkl')

    lab, x, y = [], [], []
    for name in os.listdir(direc):
        with open(direc+'/'+name,'rb') as fid:
            lab.append(name)
            label, feature_frame, feature = pickle.load(fid)
            x.append(feature)
            y.append(label)

    data = []
    data.append(x)
    data.append(y)
    data.append(lab)
    with open(direc+'/emb.pkl','wb') as fi:
        pickle.dump(data, fi, protocol=pickle.HIGHEST_PROTOCOL)

    X = np.array(x)

    # UMAP
    print('Processing UMAP')
    plt.figure(1)
    reducer = UMAP()
    U_embedded = reducer.fit_transform(X)
    U_embedded.shape
    plt.title('UMAP')
    plt.scatter(U_embedded[:,0],
                U_embedded[:,1],
                c=[palette[i] for i in y])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.grid(True)
    plt.ylim((-5,15))
    plt.xlim((-10,20))
    plt.savefig(dirout+'/UMAP_embeddings.png')


    # TSNE
    print('Processing TSNE')
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', n_jobs=4).fit_transform(X)
    print(X_embedded.shape)
    plt.figure(2)
    plt.title('TSNE')
    plt.scatter(X_embedded[:,0],
                X_embedded[:,1],
                c=[palette[i] for i in y],
                label=lab)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.grid(True)
    plt.ylim((-5,15))
    plt.xlim((-10,20))
    plt.savefig(dirout+'/TSNE_embeddings.png')


#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        print('\nDescription: Visualization with TSNE and UMAP \n')
        print('Usage: compute_umap_tsne.py folder')
        print(' folder: Folder with embeddings ')
        print('Example Dir: python compute_umap_tsne.py result/downstream/test/embeddings\n')
    else:
        main(args[0])


