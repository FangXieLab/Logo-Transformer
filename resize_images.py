import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Logo-2K+")  # "Flickr 8K"
    parser.add_argument('--brand_name', type=str, default="All")
    parser.add_argument('--ImgResol', type=int, nargs="+", default=[32])
    parser.add_argument('--clusterN', type=int, nargs="+", default=[5])
    args = parser.parse_args()

    for ImgResol in args.ImgResol:
        isExist = os.path.exists(f"./results/ImageSize{ImgResol}/fid_values")
        if not isExist:
            os.makedirs(f"./results/ImageSize{ImgResol}/fid_values")

        for clusterN in args.clusterN:
            transform = T.Resize((ImgResol, ImgResol))

            isExist = os.path.exists(f"./results/ImageSize{ImgResol}/generating_imgs/true_imgs_{args.brand_name}_clusterN_{clusterN}")
            if not isExist:
                os.makedirs(f"./results/ImageSize{ImgResol}/generating_imgs/true_imgs_{args.brand_name}_clusterN_{clusterN}")
                # os.makedirs(f"./results/ImageSize{ImgResol}/generating_imgs/generate_imgs_{args.brand_name}")

            count = 0
            brand_name_list = []
            cluster_list = []
            if args.dataset == "Famous_Brand_Logos":
                if args.brand_name == "All":
                    with open(f"./{args.dataset}/ImageSize{ImgResol}/{clusterN}Clusters/testing_data.txt") as inFile:
                        for line in inFile:
                            line = line.strip().split(",")
                            brand_name_list.append(line[2])
                            cluster_list.append(int(line[1]))
                            img_data = Image.open(f"./{args.dataset}/{line[2]}/{line[0]}").convert('RGB')
                            img_data = np.array(transform(img_data))
                            plt.imshow(img_data)
                            plt.axis('off')
                            plt.savefig(f'./results/ImageSize{ImgResol}/generating_imgs/true_imgs_{args.brand_name}_clusterN_{clusterN}/{line[2]} logo{count}.jpg',
                                bbox_inches="tight")
                            count += 1
                else:
                    with open(f"./{args.dataset}/ImageSize{ImgResol}/{clusterN}Clusters/testing_data.txt") as inFile:
                        for line in inFile:
                            line = line.strip().split(",")
                            if line[2] == args.brand_name:
                                brand_name_list.append(line[2])
                                cluster_list.append(int(line[1]))
                                img_data = Image.open(f"./{args.dataset}/{line[2]}/{line[0]}").convert('RGB')
                                img_data = np.array(transform(img_data))
                                plt.imshow(img_data)
                                plt.axis('off')
                                plt.savefig(f'./results/ImageSize{ImgResol}/generating_imgs/true_imgs_{args.brand_name}_clusterN_{clusterN}/{line[2]} logo{count}.jpg',
                                    bbox_inches="tight")
                                count += 1

            if args.dataset == "Logo-2K+":
                if args.brand_name == "All":
                    with open(f"./{args.dataset}/ImageSize{ImgResol}/{clusterN}Clusters/testing_data.txt") as inFile:
                        for line in inFile:
                            line = line.strip().split(",")
                            brand_name_list.append(line[3])
                            cluster_list.append(int(line[1]))
                            img_data = Image.open(f"./{args.dataset}/{line[2]}/{line[3]}/{line[0]}").convert('RGB')
                            img_data = np.array(transform(img_data))
                            plt.imshow(img_data)
                            plt.axis('off')
                            plt.savefig(f'./results/ImageSize{ImgResol}/generating_imgs/true_imgs_{args.brand_name}_clusterN_{clusterN}/{line[3]} logo{count}.jpg',
                                bbox_inches="tight")
                            count += 1
                            if count >= 400:
                                break
                else:
                    with open(f"./{args.dataset}/ImageSize{ImgResol}/{clusterN}Clusters/testing_data.txt") as inFile:
                        for line in inFile:
                            line = line.strip().split(",")
                            if line[3] == args.brand_name:
                                brand_name_list.append(line[3])
                                cluster_list.append(int(line[1]))
                                img_data = Image.open(f"./{args.dataset}/{line[2]}/{line[3]}/{line[0]}").convert('RGB')
                                img_data = np.array(transform(img_data))
                                plt.imshow(img_data)
                                plt.axis('off')
                                plt.savefig(f'./results/ImageSize{ImgResol}/generating_imgs/true_imgs_{args.brand_name}_clusterN_{clusterN}/{line[3]} logo{count}.jpg',
                                    bbox_inches="tight")
                                count += 1

            with open(f"./results/ImageSize{ImgResol}/generating_imgs/brand_name_list_{args.brand_name}_clusterN_{clusterN}", "wb") as fp1:  # Pickling
                # print(brand_name_list)
                pickle.dump(brand_name_list, fp1)
            with open(f"./results/ImageSize{ImgResol}/generating_imgs/cluster_list_{args.brand_name}_clusterN_{clusterN}", "wb") as fp2:  # Pickling
                pickle.dump(cluster_list, fp2)

    return 0

if __name__ == '__main__':
    main()