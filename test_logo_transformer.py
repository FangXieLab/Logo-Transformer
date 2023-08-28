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
import pandas as pd
from matplotlib.patches import Circle

from sklearn.feature_extraction.text import CountVectorizer # This allows us to specify the length of the keywords and make them into keyphrases. It also is a nice method for quickly removing stop words.
# from sentence_transformers import SentenceTransformer  # allows to quickly create high-quality embeddings that work quite well for sentence- and document-level embeddings
from sklearn.metrics.pairwise import cosine_similarity

from Logo_Transformer import Transformer

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

def prepare_heatmap_self(attn_matrix, seqN):
    data = pd.DataFrame(attn_matrix)
    index_name = {}
    for i in range(seqN):
        index_name[i] = i+1
    data.rename(index=index_name, inplace=True)
    data.index.names = ['Pixels for channels']
    data.columns.names = ['Pixels for channels']
    # data = data.T
    #data_corr.head()
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Famous_Brand_Logos") #"Flickr 8K"
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
    # assert args.batchSize == 1
    # assert args.epochs == 1
    #logger = logging.getLogger()
    #logger = get_logger('info')
    #logging.basicConfig(filename='results_captioning_images/transformer_logs/example.log', level=logging.DEBUG, filemode='w')

    output_file = ''
    for k in vars(args):
        output_file += '_'
        output_file += k
        output_file += '_'
        output_file += str(vars(args)[k])

    sys.stdout = Logger(f'results/ImageSize{args.ImgResol}/testing_logs/test{output_file}.txt')
    #tb_logger = tensorboardX.SummaryWriter('results_captioning_images/transformer_logs')
    #print(f"Hyper-parameters: {vars(args)}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: {}".format(device))
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
    state = torch.load(f"./results/ImageSize{args.ImgResol}/models/model{output_file}.pth", map_location=device)
    model.load_state_dict(state)
    model = model.to(device=device)

    # create tokenizer for brand name characters
    brand_names = ''.join(open(f"./{args.dataset}/ImageSize{args.ImgResol}/{args.clusterN}Clusters/brand_names.txt", "r").read().strip().split("\n"))
    char_vocabulary = list(set(list(brand_names)))
    stoi = dict([(x, i) for i,x in enumerate(char_vocabulary)])
    charvocab_size = len(char_vocabulary)+2+args.clusterN

    print("Start Testing...")
    step = 0
    testBatchLoss = []
    testBatchImgCE = []
    testBatchClusterCE = []
    testBatchPixAcc = []
    testBatchClusterAcc = []
    examplesN = 0
    start_epoch_time = time.time()
    batches = list(LoadLogoData(args.dataset, batchSize, "testing", args.ImgResol, args.clusterN))
    print(f"Number of testing samples: {len(batches)}")

    random.seed(10)
    show_indices = random.sample(list(np.arange(len(batches))), Show) #[random.randint(0, len(batches) - 1) for _ in range(Show)]
    show_img_test = []
    show_img_pred = []
    show_img_labels = []
    show_img_pred_labels = []
    show_names = []
    show_files = []
    #print("show_indices:", show_indices)

    attn_self_deccoder_layers = {}
    attn_encoder_deccoder_layers = {}
    for layer in range(args.n_layers):
        attn_self_deccoder_layers[layer] = []
        attn_encoder_deccoder_layers[layer] = []

    for i in range(len(batches)):
        batch = batches[i]
        img_batch = []
        label_batch = []
        cluster_label_batch = []
        name_batch = []
        file_batch = []
        if args.dataset == 'Famous_Brand_Logos':
            for imgFile, cluster_label, img_label in batch:
                #print("imgFile, caption:", imgFile, caption)
                img = Image.open(f"./{args.dataset}/{img_label}/{imgFile}").convert('RGB')
                resized_img = transform(img)
                img_data = np.array(resized_img)  # images may have different sizes (width and height)
                img_batch.append(torch.LongTensor(img_data).to(device=device))

                label_data = [int(cluster_label)+2] + [numerify(char, stoi, args.clusterN) for char in img_label]
                label_batch.append(label_data)
                cluster_label_batch.append(int(cluster_label))
                name_batch.append(img_label)
                file_batch.append(imgFile)
        elif args.dataset == 'Logo-2K+':
            for imgFile, cluster_label, cate_label, img_label in batch:
                #print("imgFile, caption:", imgFile, caption)
                # if img_label.find(' _ ') >= 0:
                #     img_label_format = img_label[:img_label.find('_')]+"&"+img_label[img_label.find('_')+1:]
                #     img = Image.open(f"./{args.dataset}/{cate_label}/{img_label_format}/{imgFile}").convert('RGB')
                # elif img_label.find('_') >= 0:
                #     img_label_format = img_label[:img_label.find('_')]+"'"+img_label[img_label.find('_')+1:]
                #     img = Image.open(f"./{args.dataset}/{cate_label}/{img_label_format}/{imgFile}").convert('RGB')
                # else:
                img = Image.open(f"./{args.dataset}/{cate_label}/{img_label}/{imgFile}").convert('RGB')
                resized_img = transform(img)
                img_data = np.array(resized_img)  # images may have different sizes (width and height)
                img_batch.append(torch.LongTensor(img_data).to(device=device))

                label_data = [int(cluster_label)+2] + [numerify(char, stoi, args.clusterN) for char in img_label]
                label_batch.append(label_data)
                cluster_label_batch.append(int(cluster_label))
                name_batch.append(img_label)
                file_batch.append(imgFile)

        img_batch = torch.cat(img_batch)
        img_batch = img_batch.view(batchSize, img_batch.shape[0] // batchSize, img_batch.shape[1],
                                       img_batch.shape[2])

        # pad label data
        label_length = max([len(x) for x in label_batch])
        for label in label_batch:
            while len(label) < label_length:
                label.append(PAD)
        label_batch = torch.LongTensor(label_batch).to(device=device)
        cluster_label_batch = torch.LongTensor(cluster_label_batch).to(device=device)
        # print("img_batch shape:", img_batch.shape)
        # print("label_batch:", label_batch)
        # print("cluster_label_batch:", cluster_label_batch)

        model.eval()

        #scheduler.step()
        #optimizer.zero_grad()
        with torch.no_grad():
            img_pred, cluster_pred = model(label_batch, img_batch, attn_visual=True, output_file=output_file)
            img_ce = model.img_ce_loss(img_pred, img_batch)
            cluster_ce = model.cluster_ce_loss(cluster_pred, cluster_label_batch)
            pixel_acc = model.pixel_accuracy(img_pred, img_batch)
            cluster_acc = model.cluster_accuracy(cluster_pred, cluster_label_batch)

            loss = model.loss(img_ce, cluster_ce, args.loss_weight)
            #loss = loss.view(loss.shape[0], -1).sum(1)
            loss = loss.mean(0)
            img_ce = img_ce.mean(0)
            cluster_ce = cluster_ce.mean(0)
            #print("loss:", loss)

            testBatchLoss.append(float(loss)*batchSize)
            testBatchImgCE.append(float(img_ce) * batchSize)
            testBatchClusterCE.append(float(cluster_ce) * batchSize)
            testBatchPixAcc.append(float(pixel_acc) * batchSize)
            testBatchClusterAcc.append(float(cluster_acc) * batchSize)
            examplesN += batchSize
        step += 1

        if i in show_indices:
            # show details for one sample
            '''print("Ground Truth Image:", img_batch[0])

            # print(img_batch.shape)
            plt.imshow(torch.Tensor.cpu(img_batch[0]))
            plt.savefig(f'./results/test_image_{args.output_file}.png')
            print("Ground Truth Cluster Label:", cluster_label_batch[0])

            print("Generated Image:", torch.argmax(img_pred[0], dim=3))
            # print(img_pred.shape)
            plt.imshow(torch.Tensor.cpu(torch.argmax(img_pred[0], dim=3)))
            plt.savefig(f'./results/pred_image_{args.output_file}.png')'''
            show_img_test.append(torch.Tensor.cpu(img_batch[0]))
            show_img_pred.append(torch.argmax(img_pred[0], dim=3))
            show_img_labels.append(torch.Tensor.cpu(cluster_label_batch[0]).numpy())
            show_img_pred_labels.append(torch.Tensor.cpu(torch.argmax(cluster_pred[0])).numpy())
            show_names.append(name_batch[0])
            show_files.append(file_batch[0])

            attn_self_decoder = np.load(f'./results/ImageSize{args.ImgResol}/attention_matrix/attention_matrix_self_decoder{output_file}.npy')
            attn_encoder_decoder = np.load(f'./results/ImageSize{args.ImgResol}/attention_matrix/attention_matrix_encoder_decoder{output_file}.npy')
            # print("attn_encoder_decoder:", attn_encoder_decoder.shape, attn_encoder_decoder)
            for layer, attn in enumerate(attn_self_decoder):
                # print("attn[0]:", attn[0].shape, attn[0].sum(1))
                attn_self_deccoder_layers[layer].append(attn[0])
            for layer, attn in enumerate(attn_encoder_decoder):
                # print("attn[0]:", attn[0].shape, attn[0].sum(1))
                attn_encoder_deccoder_layers[layer].append(attn[0])

    print('Loss: {:.3f}, image CE-loss: {:.3f}, cluster CE-loss: {:.3f}, image acc: {:.3f}, cluster acc: {:.3f}'
          .format(sum(testBatchLoss) / examplesN, sum(testBatchImgCE) / examplesN, sum(testBatchClusterCE) / examplesN,
                  sum(testBatchPixAcc) / examplesN, sum(testBatchClusterAcc) / examplesN))
    print("Processing Time: {:.2f} sec".format(time.time() - start_epoch_time))
    print()

    # show visualization of examples of testing results
    fig = plt.figure(figsize=(3*Show, 6))
    columns = Show
    rows = 2
    for show in range(Show):
        fig.add_subplot(rows, columns, show + 1)
        plt.imshow(show_img_test[show].cpu())
        plt.axis('off')
        if show == 0:
            plt.title(f"Testing samples: cluster {show_img_labels[show]}")
        else:
            plt.title(f"cluster {show_img_labels[show]}")

        fig.add_subplot(rows, columns, show + Show + 1)
        plt.imshow(show_img_pred[show].cpu())
        plt.axis('off')
        if show == 0:
            plt.title(f"Predicted samples: cluster {show_img_pred_labels[show]}")
        else:
            plt.title(f"cluster {show_img_pred_labels[show]}")
    plt.savefig(f'./results/ImageSize{args.ImgResol}/testing_imgs/test_pred_image{output_file}.png', bbox_inches="tight")

    # show heatmap of decoder self attention matrix
    fig = plt.figure(figsize=(3.5*Show, 3.0*args.n_layers))
    columns = Show
    rows = args.n_layers
    seqN = args.ImgResol**2 * args.channels
    for show in range(Show):
        for layer in range(args.n_layers):
            fig.add_subplot(rows, columns, show + 1 + layer*Show)
            # data = prepare_heatmap_self(attn_self_deccoder_layers[layer][show], seqN)
            # plt.imshow(show_img_test[show])
            # print("attn_self_deccoder_layers[layer][show]:", attn_self_deccoder_layers[layer][show].shape, attn_self_deccoder_layers[layer][show])
            # sns.set(font_scale=1.2)
            # sns.set_context({"figure.figsize": (6, 6)})
            # sns.heatmap(data=data, annot=True, linewidths=0.3, vmin=0, vmax=1, cmap="RdBu_r", center=0.5)
            data = prepare_heatmap_self(attn_self_deccoder_layers[layer][show], seqN)
            # torch.set_printoptions(profile="full")
            # np.set_printoptions(threshold=sys.maxsize)
            # print("data:", torch.Tensor(data))
            # print("data:", np.array(data))
            # mask = np.zeros_like(data)
            # mask[np.triu_indices_from(mask,1)] = True
            # sns.heatmap(data=data, mask=mask, vmin=0, vmax=1, square=True,  cmap="Blues")
            sns.heatmap(data=data, mask=(data==0), vmin=0, vmax=1, square=True,  cmap="Blues")#, annot = True, annot_kws={"size": 80 / np.sqrt(len(data))})
            # data = np.ma.array(data, mask=(data==0))
            # plt.imshow(data, interpolation=None, aspect='equal', vmin=0, vmax=1,  cmap="Blues")#, annot = True, annot_kws={"size": 80 / np.sqrt(len(data))})

            plt.axis('off')
            if layer == 0:
                plt.title(f"Samples {show+1}: layer {layer+1}")
            else:
                plt.title(f"layer {layer+1}")
    plt.tight_layout()
    plt.savefig(f'./results/ImageSize{args.ImgResol}/attention_analysis/attention_self_decoder{output_file}.png', bbox_inches="tight")

    # show visualization of decoder cross attention matrix for the first character
    fig = plt.figure(figsize=(3 * Show, 3 * args.n_layers))
    columns = Show
    rows = args.n_layers
    # fig, axs = plt.subplots(rows, columns, figsize=(20, 3 * args.n_layers))
    for show in range(Show):
        for layer in range(args.n_layers):
            axs = fig.add_subplot(rows, columns, show + 1 + layer * Show)
            attn_matrx = attn_encoder_deccoder_layers[layer][show].reshape(len(attn_encoder_deccoder_layers[layer][show]), args.ImgResol, args.ImgResol, args.channels)
            attn_matrx = attn_matrx.mean(-1)[1]
            # data = prepare_heatmap_self(attn_matrx[1]) # for the first char-pixels attention
            # print("layer:", layer, "show:", show)
            axs.imshow(show_img_test[show])
            # sns.set(font_scale=1.2)
            # sns.set_context({"figure.figsize": (6, 6)})
            # sns.heatmap(data=data, annot=True, linewidths=0.3, vmin=0, vmax=1, cmap="RdBu_r", center=0.5)
            # print("attn_matrx:", attn_matrx.shape, attn_matrx)
            for h in range(len(attn_matrx)):
                for w in range(len(attn_matrx[0])):
                    radius_size = attn_matrx[h][w] * 10
                    axs.add_patch(Circle((h, w), radius=radius_size, color='red'))

            axs.axis('off')
            if layer == 0:
                axs.set_title(f"{show_names[show]}: layer {layer + 1}")
            else:
                axs.set_title(f"layer {layer + 1}")
    plt.savefig(f'./results/ImageSize{args.ImgResol}/attention_analysis/attention_encoder_decoder_fchar{output_file}.png', bbox_inches="tight")

    
    # show visualization of decoder cross attention matrix for all characters
    charN = len(attn_encoder_deccoder_layers[0][0])
    fig = plt.figure(figsize=(3 * charN, 3 * args.n_layers))
    columns = charN
    rows = args.n_layers
    # fig, axs = plt.subplots(rows, columns, figsize=(20, 3 * args.n_layers))
    for c in range(charN):
        for layer in range(args.n_layers):
            show = 0
            # print(c, layer, c + 1 + layer * c)
            axs = fig.add_subplot(rows, columns, c + 1 + layer * charN)

            # print("attn_encoder_deccoder_layers[layer][show]:", attn_encoder_deccoder_layers[layer][show].shape, attn_encoder_deccoder_layers[layer][show].sum(-1))
            attn_matrx = attn_encoder_deccoder_layers[layer][show].reshape(
            len(attn_encoder_deccoder_layers[layer][show]), args.ImgResol, args.ImgResol, args.channels)
            attn_matrx = attn_matrx.mean(-1)[c]
            axs.imshow(show_img_test[show])
            for h in range(len(attn_matrx)):
                for w in range(len(attn_matrx[0])):
                    radius_size = attn_matrx[h][w] * 10
                    # radius_size = 0.05
                    axs.add_patch(Circle((h, w), radius=radius_size, color='red'))

            axs.axis('off')
            if c == 0:
                axs.set_title(f"{show_img_labels[show]}: layer {layer + 1}")
            else:
                axs.set_title(f"{show_names[show][c-1]}: layer {layer + 1}")
    plt.savefig(f'./results/ImageSize{args.ImgResol}/attention_analysis/attention_encoder_decoder_allchars{output_file}.png', bbox_inches="tight")

    # show visualization of decoder cross attention matrix for all characters for R, G, B channels
    cmap_options = ['Reds', 'Greens', 'Blues']
    for channel in [0, 1, 2]:
        charN = len(attn_encoder_deccoder_layers[0][0])
        fig = plt.figure(figsize=(3 * charN, 3 * args.n_layers))
        columns = charN
        rows = args.n_layers
        # fig, axs = plt.subplots(rows, columns, figsize=(20, 3 * args.n_layers))
        for c in range(charN):
            for layer in range(args.n_layers):
                show = 0
                axs = fig.add_subplot(rows, columns, c + 1 + layer * charN)
                # print("attn_encoder_deccoder_layers[layer][show]:", attn_encoder_deccoder_layers[layer][show].shape, attn_encoder_deccoder_layers[layer][show].sum(-1))
                attn_matrx = attn_encoder_deccoder_layers[layer][show].reshape(
                len(attn_encoder_deccoder_layers[layer][show]), args.ImgResol, args.ImgResol, args.channels)
                attn_matrx = attn_matrx[c,:,:,channel]
                axs.imshow(show_img_test[show][:, :, channel], cmap=cmap_options[channel])
                for h in range(len(attn_matrx)):
                    for w in range(len(attn_matrx[0])):
                        radius_size = attn_matrx[h][w] * 10
                        # radius_size = 0.05
                        axs.add_patch(Circle((h, w), radius=radius_size, color='red'))

                axs.axis('off')
                if c == 0:
                    axs.set_title(f"{show_img_labels[show]}: layer {layer + 1}")
                else:
                    axs.set_title(f"{show_names[show][c - 1]}: layer {layer + 1}")
        plt.savefig(
        f'./results/ImageSize{args.ImgResol}/attention_analysis/attention_encoder_decoder_channel{channel}{output_file}.png',
        bbox_inches="tight")




    return 0


if __name__ == '__main__':
    main()
