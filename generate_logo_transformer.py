import argparse
import time
import numpy as np
import random
from PIL import Image
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
#import tensorboardX
import logging
from tqdm import tqdm
import pickle

from sklearn.feature_extraction.text import CountVectorizer # This allows us to specify the length of the keywords and make them into keyphrases. It also is a nice method for quickly removing stop words.
# from sentence_transformers import SentenceTransformer  # allows to quickly create high-quality embeddings that work quite well for sentence- and document-level embeddings
from sklearn.metrics.pairwise import cosine_similarity

from Logo_Transformer import Transformer
from Logo_Transformer import Generator

PAD = 0
OOV = 1
Show = 10
#batchSize = 4
#captionNum = 5
#n_gram = 1
#top_n = 5
#diversity_rate = 0.7
#ImgResol = 8
#epochs = 5
#loss_weight = 0.5
#split_rate = 0.0013

def numerify(token, stoi, clusterN):
    if token not in stoi:
        return OOV
    else:
        return stoi[token]+2+clusterN  # 2~2+N: clusters

def LoadLogoData(dataset, batchSize, partition, ImgResol, clusterN):
    assert partition in ["training", "validation", "testing"]
    with open(f"./{dataset}/ImageSize{ImgResol}/{clusterN}Clusters/{partition}_data.txt", "rb") as inFile:
        while True:
            buff = []
            for _ in range(batchSize):
                try:
                    buff.append(next(inFile).decode().strip().split(","))
                except StopIteration:
                    break

            #print("buff len:", len(buff))
            if len(buff) < batchSize:
                break
            random.shuffle(buff)

            if dataset == 'Famous_Brand_Logos':
                for instance in buff:
                    imgFile = instance[0]
                    cluster_label = instance[1]
                    img_label = instance[2]
            elif dataset == 'Logo-2K+':
                for instance in buff:
                    imgFile = instance[0]
                    cluster_label = instance[1]
                    cate_label = instance[2]
                    img_label = instance[3]
            #r = [(imgFile, cluster_label, img_label)]
            r = buff
            #print("r len:", len(r))
            if len(r) > 0:
                yield r

def LoadCaptioningImage(batchSize, captionNum, partition):
    assert partition in ["training", "validation", "testing"]
    with open(f"./Flickr 8K/captions_{partition}.txt", "rb") as inFile:
        while True:
            buff = []
            for _ in range(captionNum * batchSize):
                try:
                    buff.append(next(inFile).decode().strip().split(","))
                except StopIteration:
                    break

            #print("buff len:", len(buff))
            if len(buff) < batchSize * captionNum: #== 0:
                break
            concatenated_caption = {}
            for instance in buff:
                imgFile = instance[0]
                caption = instance[1]
                if imgFile not in concatenated_caption:
                    concatenated_caption[imgFile] = ''
                concatenated_caption[imgFile] += ' ' + caption
            imgKeys = list(concatenated_caption.keys())  # random.shuffle(concatenated_caption)
            random.shuffle(imgKeys)
            r = [(imgFile, concatenated_caption[imgFile]) for imgFile in imgKeys]
            #print("r len:", len(r))
            if len(r) > 0:
                yield r

def train_validation_split(folder_name, data, captionNum, split_rate):
    captions = open(f"./{folder_name}/{data}.txt", "r").readlines()
    rows = len(captions)
    assert rows % captionNum == 0
    split = int(rows * split_rate) - int(rows * split_rate) % captionNum
    split2 = int(split * 0.65) - int(split * 0.65) % captionNum
    training_data = captions[:split2]
    validation_data = captions[split2:split]
    testing_data = captions[split:]
    with open(f"./{folder_name}/{data}_training.txt", "w") as outFile:
        outFile.writelines(training_data)
    with open(f"./{folder_name}/{data}_validation.txt", "w") as outFile:
        outFile.writelines(validation_data)
    with open(f"./{folder_name}/{data}_testing.txt", "w") as outFile:
        outFile.writelines(testing_data)
    return training_data, validation_data, testing_data

def mmr(doc_embedding, word_embeddings, words, top_n, diversity):
    '''
    We start by selecting the keyword/keyphrase that is the most similar to the document.
    Then, we iteratively select new candidates that are both similar to the document and
    not similar to the already selected keywords/keyphrases.

    '''
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        # in terms of all current keywords, get the max similarities
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx], [word_embeddings[idx] for idx in keywords_idx]

def keywords_extraction(n_gram, text, top_n, diversity_rate):
    n_gram_range = (n_gram, n_gram)  # lower bound and upper bound of word length (after stop words removal)
    stop_words = "english"

    # Extract candidate words/phrases
    # remove stop words and keep only one of the duplicates
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
    candidates = count.get_feature_names_out()

    # embedding by pre-trained BERT model
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)

    # get keywords based on the highest similarities, but there could be high similarities between keywords as well
    # distances = cosine_similarity(doc_embedding, candidate_embeddings)
    # keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    keywords, keywords_embed = mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity_rate)
    return keywords, keywords_embed

def get_lr(step, args):
    warmup_steps = args.warmup
    lr_base = args.lr * 0.002 # for Adam correction
    ret = 5000. * args.hidden_size ** (-0.5) * \
          np.min([(step + 1) * warmup_steps ** (-1.5), (step + 1) ** (-0.5)])
    return ret * lr_base

def get_logger(LEVEL, log_file = None):
    head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
    if LEVEL == 'info':
        logging.basicConfig(level=logging.INFO, format=head)
    elif LEVEL == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    if log_file != None:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    return logger

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
    parser.add_argument('--brand_name', type=str, default="Google")
    parser.add_argument('--dataset', type=str, default="Famous_Brand_Logos") #"Flickr 8K"
    # parser.add_argument('--cluster_required', type=int, default=0)
    # parser.add_argument('--model_file', type=str, default="default")
    # parser.add_argument('--output_file', type=str, default="default")
    parser.add_argument('--ImgResol', type=int, default=2)
    parser.add_argument('--clusterN', type=int, default=5)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--channels', type=int, default=3)
    # parser.add_argument('--n_gram', type=int, default=1)
    # parser.add_argument('--top_n', type=int, default=5)
    # parser.add_argument('--diversity_rate', type=float, default=0.7)
    parser.add_argument('--loss_weight', type=float, default=0.1)
    # parser.add_argument('--split_rate', type=float, default=0.7)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--filter_size', type=int, default=128)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--head_size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--att_type', type=str, default="local_1d")
    parser.add_argument('--block_length', type=int, default=10)

    args = parser.parse_args()
    batchSize = 1
    # assert args.cluster_required in list(np.arange(args.clusterN))
    #logger = logging.getLogger()
    #logger = get_logger('info')
    #logging.basicConfig(filename='results_captioning_images/transformer_logs/example.log', level=logging.DEBUG, filemode='w')
    #sys.stdout = Logger(f'results_wrong_mask/logs/test_{args.output_file}.txt')
    #tb_logger = tensorboardX.SummaryWriter('results_captioning_images/transformer_logs')
    #print(f"Hyper-parameters: {vars(args)}")

    output_file = ''
    for k in vars(args):
        output_file += '_'
        output_file += k
        output_file += '_'
        output_file += str(vars(args)[k])
    output_file = output_file[output_file.find(args.brand_name):]
    model_file = output_file[output_file.find("dataset")-1:]

    isExist = os.path.exists(f"./results/ImageSize{args.ImgResol}/generating_imgs/generate_imgs_{output_file}")
    if not isExist:
        os.makedirs(f"./results/ImageSize{args.ImgResol}/generating_imgs/generate_imgs_{output_file}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print("Using device: {}".format(device))
    #logging.info("Using device: {}".format(device))

    # split dataset
    # training_data, validation_data, testing_data = train_validation_split(args.dataset, "captions", captionNum, args.split_rate)
    # print(f"Number of training samples: {int(len(training_data)/captionNum)}")
    # print(f"Number of validation samples: {int(len(validation_data)/captionNum)}")
    # print(f"Number of testing samples: {int(len(testing_data) / captionNum)}")

    transform = T.Resize((args.ImgResol, args.ImgResol))
    # build model
    model = Transformer(#charvocab_size=args.charvocab_size,#100
                        dataset=args.dataset,
                        clusterN=args.clusterN,
                        n_layers=args.n_layers,  # 6
                        hidden_size=args.hidden_size,  # 512
                        filter_size=args.filter_size,  # 2048
                        dropout_rate=args.dropout_rate,
                        head_size=args.head_size,
                        att_type=args.att_type,
                        block_length=args.block_length,
                        has_inputs=True,
                        src_pad_idx=PAD,
                        trg_pad_idx=None,
                        distr="cat",
                        channels=args.channels,
                        img_size=args.ImgResol)
    state = torch.load(f"./results/ImageSize{args.ImgResol}/models/model{model_file}.pth", map_location=device)
    model.load_state_dict(state)
    model = model.to(device=device)
    generator = Generator(inferencing_model=model)

    # create tokenizer for brand name characters
    brand_names = ''.join(open(f"./{args.dataset}/ImageSize{args.ImgResol}/{args.clusterN}Clusters/brand_names.txt", "r").read().strip().split("\n"))
    char_vocabulary = list(set(list(brand_names)))
    stoi = dict([(x, i) for i,x in enumerate(char_vocabulary)])
    charvocab_size = len(char_vocabulary)+2+args.clusterN

    # get initial pixels for clusters
    # initial_pixel = {}
    # count_dict = {}
    # for i in range(args.clusterN):
    #     initial_pixel[i] = torch.zeros((1,args.channels)).type(torch.LongTensor).unsqueeze(0)
    #     count_dict[i] = 0
    #print("initial_pixel:", initial_pixel)

    # isExist = os.path.exists(f"./results/{args.dataset}/true_imgs_{args.brand_name}")
    # if not isExist:
    #     os.makedirs(f"./results/{args.dataset}/true_imgs_{args.brand_name}")
    #     os.makedirs(f"./results/{args.dataset}/generate_imgs_{args.brand_name}")

    # count = 0
    # brand_name_list = []
    # cluster_list = []
    # if args.dataset == "Famous_Brand_Logos":
    #     if args.brand_name == "All":
    #         with open(f"./{args.dataset}/ImageSize{args.ImgResol}/training_data.txt") as inFile:
    #             for line in inFile:
    #                 line = line.strip().split(",")
    #                 brand_name_list.append(line[2])
    #                 cluster_list.append(int(line[1]))
    #                 img_data = Image.open(f"./{args.dataset}/{line[2]}/{line[0]}").convert('RGB')
    #                 img_data = np.array(transform(img_data))
    #                 plt.imshow(img_data)
    #                 plt.axis('off')
    #                 plt.savefig(f'./results/{args.dataset}/true_imgs_{args.brand_name}/{line[2]} logo{count}.jpg',
    #                     bbox_inches="tight")
    #                 img_data = torch.FloatTensor(img_data).to(device=device)
    #
    #                 initial_pixel[int(line[1])] += torch.mean(img_data.view(args.ImgResol * args.ImgResol, args.channels),0).type(torch.LongTensor)
    #                 # print("initial_pixel[int(line[1])]:", initial_pixel[int(line[1])])
    #                 count_dict[int(line[1])] += 1
    #                 count += 1
    #     else:
    #         with open(f"./{args.dataset}/ImageSize{args.ImgResol}/training_data.txt") as inFile:
    #             for line in inFile:
    #                 line = line.strip().split(",")
    #                 if line[2] == args.brand_name:
    #                     brand_name_list.append(line[2])
    #                     cluster_list.append(int(line[1]))
    #
    #                     img_data = Image.open(f"./{args.dataset}/{line[2]}/{line[0]}").convert('RGB')
    #                     img_data = np.array(transform(img_data))
    #                     plt.imshow(img_data)
    #                     plt.axis('off')
    #                     plt.savefig(f'./results/{args.dataset}/true_imgs_{args.brand_name}/{line[2]} logo{count}.jpg',
    #                         bbox_inches="tight")
    #                     img_data = torch.FloatTensor(img_data).to(device=device)
    #
    #                     initial_pixel[int(line[1])] += torch.mean(img_data.view(args.ImgResol * args.ImgResol, args.channels),0).type(torch.LongTensor)
    #                     # print("initial_pixel[int(line[1])]:", initial_pixel[int(line[1])])
    #                     count_dict[int(line[1])] += 1
    #                     count += 1

    # print("initial_pixel:", initial_pixel)
    # print("count_dict:", count_dict)
    # '''if args.dataset == "Logo-2K+":
    #     if args.brand_name == "All":
    #         with open(f"./{args.dataset}/ImageSize{args.ImgResol}/training_data.txt", "rb") as inFile:
    #             for line in inFile:
    #                 line = line.strip().split(",")
    #                 img_data = Image.open(f"./{args.dataset}/{line[2]}/{line[3]}/{line[0]}").convert('RGB')
    #                 img_data = torch.LongTensor(np.array(transform(img_data))).to(device=device)
    #                 initial_pixel[int(line[1])] += img_data.view(args.ImgResol*args.ImgResol, args.channels).mean(0).unsqueeze(0).unsqueeze(0)
    #                 count_dict[int(line[1])] += 1'''

    # for cluster_i in initial_pixel:
    #     initial_pixel[cluster_i] = (initial_pixel[cluster_i]/count_dict[cluster_i]).type(torch.LongTensor)
    #     # print("initial_pixel[cluster_i]:", initial_pixel[cluster_i])
    #     initial_pixel[cluster_i] = initial_pixel[cluster_i][0][0][0].unsqueeze(0).unsqueeze(0).unsqueeze(0) # obtain first channel pixel value
    #     # print("initial_pixel[cluster_i]:", initial_pixel[cluster_i])
    # print("initial_pixel:", initial_pixel)

    # with open(f"./results/ImageSize{args.ImgResol}/generating_imgs/brand_name_list_{args.brand_name}_clusterN_{args.clusterN}", "rb") as fp1:  # Pickling
    with open(f"./results/ImageSize{args.ImgResol}/generating_imgs/brand_name_list_All_clusterN_{args.clusterN}","rb") as fp1:  # Pickling
        brand_name_list = pickle.load(fp1)
    with open(f"./results/ImageSize{args.ImgResol}/generating_imgs/cluster_list_All_clusterN_{args.clusterN}", "rb") as fp2:  # Pickling
        cluster_list = pickle.load(fp2)

    print("Start Generating...")
    count = 0
    if args.brand_name == "All":
        for br, cl in zip(brand_name_list, cluster_list):
            input_data = torch.LongTensor([cl+2] + [numerify(char, stoi, args.clusterN) for char in br]).to(device=device)
            #input_data = input_data.unsqueeze(0)

            initial_pixel = None
            img_pred, cluster_pred = generator(input_data, device, initial_pixel)
            plt.imshow(img_pred.cpu())
            plt.axis('off')
            # plt.title(f"Given brand name: {args.brand_name}, cluster: {args.cluster_required}")
            plt.savefig(f'./results/ImageSize{args.ImgResol}/generating_imgs/generate_imgs_{output_file}/{br} logo{count}.jpg', bbox_inches="tight")
            count += 1
    else:
        for br, cl in zip(brand_name_list, cluster_list):
            # print(br, cl)
            if br == args.brand_name:
                input_data = torch.LongTensor([cl + 2] + [numerify(char, stoi, args.clusterN) for char in br]).to(
                    device=device)
                # input_data = input_data.unsqueeze(0)

                initial_pixel = None
                img_pred, cluster_pred = generator(input_data, device, initial_pixel)
                plt.imshow(img_pred.cpu())
                plt.axis('off')
                # plt.title(f"Given brand name: {args.brand_name}, cluster: {args.cluster_required}")
                plt.savefig(
                    f'./results/ImageSize{args.ImgResol}/generating_imgs/generate_imgs_{output_file}/{br} logo{count}.jpg',
                    bbox_inches="tight")
                count += 1

    return 0


if __name__ == '__main__':
    main()
