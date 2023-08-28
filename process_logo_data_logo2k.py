import argparse
import os
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as T
import random
# from tqdm import tqdm

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# from sentence_transformers import SentenceTransformer, util

import pickle

class Logger():
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ImgResol', type=int, default=16)
    parser.add_argument('--clusterN', type=int, nargs="+", default=[5])
    args = parser.parse_args()

    isExist = os.path.exists(f"./Logo-2K+/ImageSize{args.ImgResol}")
    if not isExist:
        # os.makedirs(f"./Logo-2K+/ImageSize{args.ImgResol}")
        for cluster in args.clusterN:
            os.makedirs(f"./Logo-2K+/ImageSize{args.ImgResol}/{cluster}Clusters")

    isExist = os.path.exists(f"./results/ImageSize{args.ImgResol}/")
    if not isExist:
        os.makedirs(f"./results/ImageSize{args.ImgResol}/clustering_logs")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/clustering_plots")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/training_logs")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/testing_logs")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/training_plots")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/testing_imgs")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/generating_imgs")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/models")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/attention_analysis")
        os.makedirs(f"./results/ImageSize{args.ImgResol}/attention_matrix")

    sys.stdout = Logger(f'results/ImageSize{args.ImgResol}/clustering_logs/cluster_ImgResol_{args.ImgResol}.txt')

    ImgResol = args.ImgResol
    img_files = []
    img_data = []
    img_embed = []
    img_cate = []
    img_labels = []
    transform = T.Resize((ImgResol, ImgResol))
    # may use image embedding to cluster
    # bert_model = SentenceTransformer('clip-ViT-B-32')

    for category in os.listdir('./Logo-2K+'):
        # print(brand)
        if (category == '.DS_Store') or (category.find("ImageSize") >= 0):
            continue
        for brand in os.listdir(f'./Logo-2K+/{category}'):
            if brand == '.DS_Store':
                continue
            for imgFile in os.listdir(f'./Logo-2K+/{category}/{brand}'):
                # print(imgFile)
                img = Image.open(f"./Logo-2K+/{category}/{brand}/{imgFile}").convert('RGB')
                # print(np.array(img).shape)
                resized_img = transform(img)
                # plt.imshow(resized_img)
                # print(np.array(resized_img).shape)
                img_files.append(imgFile)
                img_data.append(np.array(resized_img))
                img_embed.append(resized_img)
                img_cate.append(category)
                img_labels.append(brand)

    # img_emb = model.encode(Image.open('two_dogs_in_snow.jpg'))
    img_data = np.array(img_data)
    img_data = img_data.reshape(img_data.shape[0], -1)
    # img_embed = bert_model.encode(img_embed)

    # scaling to 0 mean 1 std
    # scaler = StandardScaler()
    # scaled_img_data = scaler.fit_transform(img_data)
    # scaled_img_embed = scaler.fit_transform(img_embed)

    print("Number of samples:", img_data.shape[0])
    print("Number of unique image label (brand name):", len(set(img_labels)))
    print("Image flatten data size:", img_data.shape)
    # print("Image embedding size:", img_embed.shape)

    # choose optimal number of clusters based on sse or largest silhouette score
    # use k-means++ to ensure centroids are initialized with some distance between them
    # n_init: increase the value to ensure to find a stable solution
    # max_iter: increase the value to ensure k-means can converge
    k_max = 20
    kmeans_kwargs = {
        'init': "k-means++",  # "random"
        'n_init': 50,
        'max_iter': 500}

    sse_list = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(img_data)
        sse_list.append(kmeans.inertia_)

    # A silhouette coefficient of 0 indicates that clusters are
    # significantly overlapping one another, and a silhouette coefficient
    # of 1 indicates clusters are well-separated
    silhouette_coefficients = []
    for k in range(2, k_max + 1):  # needs a minimum of 2 clusters
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(img_data)
        score = silhouette_score(img_data, kmeans.labels_)
        silhouette_coefficients.append(score)

    # use kneed to identify the elbow point
    kl = KneeLocator(range(1, k_max + 1), sse_list, curve="convex", direction="decreasing")
    print("Identified elbow point:", kl.elbow)

    plt.style.use("fivethirtyeight")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(range(1, k_max + 1), sse_list)
    axs[1].plot(range(2, k_max + 1), silhouette_coefficients)
    axs[0].set_xticks(range(1, k_max + 1))
    axs[1].set_xticks(range(1, k_max + 1))
    axs[0].set_xlabel("Number of Clusters")
    axs[1].set_xlabel("Number of Clusters")
    axs[0].set_ylabel("SSE")
    axs[1].set_ylabel("Silhouette Coefficient")
    fig.suptitle("Image RGB data for clustering")
    # plt.show()
    plt.savefig(f"./results/ImageSize{args.ImgResol}/clustering_plots/cluster_ImgResol_{args.ImgResol}_results.png", bbox_inches="tight")

    # define K-mean class,
    # nondeterministic: perform n_init times of k-mean to choose the one with lowest sse
    k_optima_list = args.clusterN
    for k_optima in k_optima_list:
        kmeans = KMeans(
        init="k-means++",
        n_clusters=k_optima,
        n_init=50,
        max_iter=500)

        kmeans.fit(img_data)  # expected dim<=2
        cluster_labels = kmeans.labels_
        print("")
        print(f"For {k_optima} clusters:")
        print("Number of cluster defined:", k_optima)
        print("The lowest SSE value:", kmeans.inertia_)
        # print("Final locations of the centroid:", kmeans.cluster_centers_)
        print("Number of iterations required to converge:", kmeans.n_iter_)
        # print("Predicted labels:", kmeans.labels_)

        exampleN = 10
        fig = plt.figure(figsize=(20, k_optima*2))
        columns = exampleN
        rows = k_optima
        for i in range(k_optima):
            # print(f"{exampleN} examples for each cluster:")
            indices = [j for j, x in enumerate(cluster_labels) if x == i]
            if len(indices) == 0:
                # print("No images belong to this cluster.")
                pass
            else:
                for show in range(exampleN):
                    fig.add_subplot(rows, columns, show + 1 + i*exampleN)
                    plt.imshow(img_data[random.choice(indices)].reshape(ImgResol, ImgResol, 3))
                    plt.axis('off')
                    plt.title(i+1)
        # plt.show()
        plt.tight_layout()
        plt.savefig(
                f"./results/ImageSize{args.ImgResol}/clustering_plots/cluster_ImgResol_{args.ImgResol}_clusterN_{k_optima}_{exampleN}examples.png",
                bbox_inches="tight")
        plt.close(fig)


        with open(f"./Logo-2K+/ImageSize{args.ImgResol}/{k_optima}Clusters/kmeans_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)

        # concatenate labels and pad, split and save as training, validation and testing set
        '''concatenated_labels = []
        for cl, il in zip(cluster_labels, img_labels):
            concatenated_labels.append(str(cl)+','+il)'''

        # pad so that all labels (as encoder input) have the same length
        '''n = len(max(concatenated_labels, key=len))
        print(f"Longest label: {max(concatenated_labels, key=len)}, length of {n}", )
        for i in range(len(concatenated_labels)):
            while len(concatenated_labels[i]) < n:
                concatenated_labels[i].append('PAD')'''

        # put padding under batches
        # shuffle and split
        combined_data = list(zip(img_files, cluster_labels, img_cate, img_labels))
        random.shuffle(combined_data)
        img_files, cluster_labels, img_cate, img_labels = zip(*combined_data)
        # img_files[0], cluster_labels[0], img_labels[0]
        # img_files.index('NIKE logo84.jpg'), cluster_labels[345], img_labels[345]

        split_rate1 = 0.7
        split_rate2 = 0.65
        with open(f"./Logo-2K+/ImageSize{args.ImgResol}/{k_optima}Clusters/training_data.txt", "w") as outFile:
            for i in range(int(len(img_files) * split_rate1 * split_rate2)):
                print(f"{img_files[i]},{cluster_labels[i]},{img_cate[i]},{img_labels[i]}", file=outFile)
        outFile.close()

        with open(f"./Logo-2K+/ImageSize{args.ImgResol}/{k_optima}Clusters/validation_data.txt", "w") as outFile:
            for i in range(int(len(img_files) * split_rate1 * split_rate2), int(len(img_files) * split_rate1)):
                print(f"{img_files[i]},{cluster_labels[i]},{img_cate[i]},{img_labels[i]}", file=outFile)
        outFile.close()

        with open(f"./Logo-2K+/ImageSize{args.ImgResol}/{k_optima}Clusters/testing_data.txt", "w") as outFile:
            for i in range(int(len(img_files) * split_rate1), len(img_files)):
                print(f"{img_files[i]},{cluster_labels[i]},{img_cate[i]},{img_labels[i]}", file=outFile)
        outFile.close()

        print("Number of training samples:", int(len(img_files) * split_rate1 * split_rate2))
        print("Number of validation samples:", int(len(img_files) * split_rate1 * (1 - split_rate2)))
        print("Number of testing samples:", int(len(img_files) * (1 - split_rate1)))

        # vocabulary should exclude data from testing set
        with open(f"./Logo-2K+/ImageSize{args.ImgResol}/{k_optima}Clusters/brand_names.txt", "w") as outFile:
            for brand_name in set(img_labels[:int(len(img_files) * split_rate1)]):
                print(f"{brand_name}", file=outFile)
        outFile.close()
    return 0

if __name__ == '__main__':
    main()
