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

from sklearn.feature_extraction.text import CountVectorizer # This allows us to specify the length of the keywords and make them into keyphrases. It also is a nice method for quickly removing stop words.
# from sentence_transformers import SentenceTransformer  # allows to quickly create high-quality embeddings that work quite well for sentence- and document-level embeddings
from sklearn.metrics.pairwise import cosine_similarity

from Logo_Transformer import Transformer

PAD = 0
OOV = 1
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
    parser.add_argument('--dataset', type=str, default="Famous_Brand_Logos") #"Flickr 8K"
    # parser.add_argument('--output_file', type=str, default="default")
    parser.add_argument('--ImgResol', type=int, default=2)
    parser.add_argument('--clusterN', type=int, default=5)
    parser.add_argument('--batchSize', type=int, default=8)
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
    #logger = logging.getLogger()
    #logger = get_logger('info')
    #logging.basicConfig(filename='results_captioning_images/transformer_logs/example.log', level=logging.DEBUG, filemode='w')

    output_file = ''
    for k in vars(args):
        output_file += '_'
        output_file += k
        output_file += '_'
        output_file += str(vars(args)[k])

    sys.stdout = Logger(f'results/ImageSize{args.ImgResol}/training_logs/train{output_file}.txt')
    #tb_logger = tensorboardX.SummaryWriter('results_captioning_images/transformer_logs')
    print(f"Hyper-parameters: {vars(args)}")

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
    model = model.to(device=device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {}".format(num_params))

    optimizer = optim.Adam(model.parameters(), lr=1., betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step, args))

    # create tokenizer for brand name characters
    brand_names = ''.join(open(f"./{args.dataset}/ImageSize{args.ImgResol}/{args.clusterN}Clusters/brand_names.txt", "r").read().strip().split("\n"))
    char_vocabulary = list(set(list(brand_names)))
    stoi = dict([(x, i) for i,x in enumerate(char_vocabulary)])
    charvocab_size = len(char_vocabulary)+2+args.clusterN

    trainEpochLoss = []
    trainEpochImgCE = []
    trainEpochClusterCE = []
    trainEpochPixAcc = []
    trainEpochClusterAcc = []
    devEpochLoss = []
    devEpochImgCE = []
    devEpochClusterCE = []
    devEpochPixAcc = []
    devEpochClusterAcc = []

    start_time = time.time()
    for epoch in range(args.epochs):
        print("----------------- Epoch:", epoch, "-----------------")
        print("Start Training...")
        step1 = 0
        trainBatchLoss = []
        trainBatchImgCE = []
        trainBatchClusterCE = []
        trainBatchPixAcc = []
        trainBatchClusterAcc = []
        examplesN1 = 0
        start_epoch_time = time.time()
        for batch in LoadLogoData(args.dataset, args.batchSize, "training", args.ImgResol, args.clusterN):
            img_batch = []
            label_batch = []
            cluster_label_batch = []
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
            elif args.dataset == 'Logo-2K+':
                for imgFile, cluster_label, cate_label, img_label in batch:
                    # print("imgFile, caption:", imgFile, caption)
                    img = Image.open(f"./{args.dataset}/{cate_label}/{img_label}/{imgFile}").convert('RGB')
                    resized_img = transform(img)
                    img_data = np.array(resized_img)  # images may have different sizes (width and height)
                    img_batch.append(torch.LongTensor(img_data).to(device=device))

                    label_data = [int(cluster_label) + 2] + [numerify(char, stoi, args.clusterN) for char in img_label]
                    label_batch.append(label_data)
                    cluster_label_batch.append(int(cluster_label))

            img_batch = torch.cat(img_batch)
            img_batch = img_batch.view(args.batchSize, img_batch.shape[0] // args.batchSize, img_batch.shape[1],
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

            model.train()

            #scheduler.step()
            optimizer.zero_grad()
            # print("image data before feeding  into model:", img_batch.shape, img_batch)
            # print("label data before feeding  into model:", label_batch.shape, label_batch)
            img_pred, cluster_pred = model(label_batch, img_batch, attn_visual=False, output_file="None")
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

            loss.backward()

            optimizer.step()
            scheduler.step()
            trainBatchLoss.append(float(loss)*args.batchSize)
            trainBatchImgCE.append(float(img_ce) * args.batchSize)
            trainBatchClusterCE.append(float(cluster_ce) * args.batchSize)
            trainBatchPixAcc.append(float(pixel_acc) * args.batchSize)
            trainBatchClusterAcc.append(float(cluster_acc) * args.batchSize)
            examplesN1 += args.batchSize
            #print(f"step: {step1}, lr: {scheduler.get_last_lr()}")
            if step1 % 100 == 0:
                print('  |Batch: {}, loss: {:.3f}, image CE-loss: {:.3f}, cluster CE-loss: {:.3f}, image acc: {:.3f}, cluster acc: {:.3f}'
                      .format(step1, loss.item(), img_ce.item(), cluster_ce.item(), pixel_acc.item(), cluster_acc.item()))

                # logging.info('  |Batch: {}, loss: {:.3f}, image acc: {:.3f}, image CE-loss: {:.3f}, keywords similarity: {:.3f}'
                #       .format(step1, loss.item(), pixel_acc.item(), img_ce.item(), kw_similarity.item()))
                # tb_logger.add_scalar('loss', loss.item())
                # tb_logger.add_scalar('pixel_acc', pixel_acc.item())
                # tb_logger.add_scalar('img_ce', img_ce.item())
                # tb_logger.add_scalar('kw_similarity', kw_similarity)

            step1 += 1
            # if step1 >= 1:
            #     print(njnn)
        trainEpochLoss.append(round(sum(trainBatchLoss)/examplesN1, 3))
        trainEpochImgCE.append(round(sum(trainBatchImgCE) / examplesN1, 3))
        trainEpochClusterCE.append(round(sum(trainBatchClusterCE) / examplesN1, 3))
        trainEpochPixAcc.append(round(sum(trainBatchPixAcc) / examplesN1, 3))
        trainEpochClusterAcc.append(round(sum(trainBatchClusterAcc) / examplesN1, 3))
        #print("Epoch Time: {:.2f} sec".format(time.time() - start_epoch_time))
        print()

        print("Start Validation...")
        step2 = 0
        devBatchLoss = []
        devBatchImgCE = []
        devBatchClusterCE = []
        devBatchPixAcc = []
        devBatchClusterAcc = []
        examplesN2 = 0
        start_epoch_time = time.time()
        for batch in LoadLogoData(args.dataset, args.batchSize, "validation", args.ImgResol, args.clusterN):
            img_batch = []
            label_batch = []
            cluster_label_batch = []
            if args.dataset == 'Famous_Brand_Logos':
                for imgFile, cluster_label, img_label in batch:
                    img = Image.open(f"./{args.dataset}/{img_label}/{imgFile}").convert('RGB')
                    resized_img = transform(img)
                    img_data = np.array(resized_img)  # images may have different sizes (width and height)
                    img_batch.append(torch.LongTensor(img_data).to(device=device))

                    label_data = [int(cluster_label)+2] + [numerify(char, stoi, args.clusterN) for char in img_label]
                    label_batch.append(label_data)
                    cluster_label_batch.append(int(cluster_label))
            elif args.dataset == 'Logo-2K+':
                for imgFile, cluster_label, cate_label, img_label in batch:
                    img = Image.open(f"./{args.dataset}/{cate_label}/{img_label}/{imgFile}").convert('RGB')
                    resized_img = transform(img)
                    img_data = np.array(resized_img)  # images may have different sizes (width and height)
                    img_batch.append(torch.LongTensor(img_data).to(device=device))

                    label_data = [int(cluster_label) + 2] + [numerify(char, stoi, args.clusterN) for char in img_label]
                    label_batch.append(label_data)
                    cluster_label_batch.append(int(cluster_label))

            img_batch = torch.cat(img_batch)
            img_batch = img_batch.view(args.batchSize, img_batch.shape[0] // args.batchSize, img_batch.shape[1],
                                       img_batch.shape[2])#.to(device=device)

            # pad label data
            label_length = max([len(x) for x in label_batch])
            for label in label_batch:
                while len(label) < label_length:
                    label.append(PAD)
            label_batch = torch.LongTensor(label_batch).to(device=device)
            cluster_label_batch = torch.LongTensor(cluster_label_batch).to(device=device)

            model.eval()
            with torch.no_grad():
                img_pred, cluster_pred = model(label_batch, img_batch, attn_visual=False, output_file="None")
                img_ce = model.img_ce_loss(img_pred, img_batch)
                cluster_ce = model.cluster_ce_loss(cluster_pred, cluster_label_batch)
                pixel_acc = model.pixel_accuracy(img_pred, img_batch)
                cluster_acc = model.cluster_accuracy(cluster_pred, cluster_label_batch)

                loss = model.loss(img_ce, cluster_ce, args.loss_weight)
                # print("loss:", loss.shape, loss)
                # loss = loss.view(loss.shape[0], -1).sum(1)
                loss = loss.mean(0)
                img_ce = img_ce.mean(0)
                cluster_ce = cluster_ce.mean(0)

                devBatchLoss.append(float(loss) * args.batchSize)
                devBatchImgCE.append(float(img_ce) * args.batchSize)
                devBatchClusterCE.append(float(cluster_ce) * args.batchSize)
                devBatchPixAcc.append(float(pixel_acc) * args.batchSize)
                devBatchClusterAcc.append(float(cluster_acc) * args.batchSize)
                examplesN2 += args.batchSize

                if step2 % 100 == 0:
                    print(
                        '  |Batch: {}, loss: {:.3f}, image CE-loss: {:.3f}, cluster CE-loss: {:.3f}, image acc: {:.3f}, cluster acc: {:.3f}'
                        .format(step2, loss.item(), img_ce.item(), cluster_ce.item(), pixel_acc.item(),
                                cluster_acc.item()))

            step2 += 1
            torch.save(model.state_dict(), f"./results/ImageSize{args.ImgResol}/models/model{output_file}.pth")
        # if (epoch == 50) or (epoch == 100) or (epoch == 150):
        #     output_file_epoch = output_file[:output_file.find('epochs')+7]+str(epoch)+output_file[output_file.find('_channels'):]
        #     # print("output_file_epoch:", output_file_epoch)
        #     torch.save(model.state_dict(), f"./results/ImageSize{args.ImgResol}/models/model{output_file_epoch}.pth")

        devEpochLoss.append(round(sum(devBatchLoss) / examplesN2, 3))
        devEpochImgCE.append(round(sum(devBatchImgCE) / examplesN2, 3))
        devEpochClusterCE.append(round(sum(devBatchClusterCE) / examplesN2, 3))
        devEpochPixAcc.append(round(sum(devBatchPixAcc) / examplesN2, 3))
        devEpochClusterAcc.append(round(sum(devBatchClusterAcc) / examplesN2, 3))
        print("Epoch Processing Time: {:.2f} sec".format(time.time() - start_epoch_time))
        print()
        # print(nkk)
    print("Total Processing Time: {:.2f} sec".format(time.time() - start_time))
    #tb_logger.close()

    metrics = ['Loss', 'ImgCE', 'ClusterCE', 'PixAcc', 'ClusterAcc']
    partitions = ['Training', 'Validation']
    results = {'Loss': {'Training': trainEpochLoss, 'Validation': devEpochLoss},
               'ImgCE': {'Training': trainEpochImgCE, 'Validation': devEpochImgCE},
               'ClusterCE': {'Training': trainEpochClusterCE, 'Validation': devEpochClusterCE},
               'PixAcc': {'Training': trainEpochPixAcc, 'Validation': devEpochPixAcc},
               'ClusterAcc': {'Training': trainEpochClusterAcc, 'Validation': devEpochClusterAcc}}
    fig, axs = plt.subplots(1,5, figsize=(20,3)) #plt.figure(figsize=(6, 25))
    #axs = axs.flatten()
    for i in range(len(metrics)):
        for j in partitions:
            data = results[metrics[i]][j]
            sns.lineplot(x=np.arange(1, args.epochs + 1), y=data, ax=axs[i], label=f'{j} {metrics[i]}')
            axs[i].set_title(f'Training {metrics[i]} vs Validation {metrics[i]}')
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(metrics[i])
            axs[i].legend()
    #plt.show()
    plt.tight_layout()
    plt.savefig(f"./results/ImageSize{args.ImgResol}/training_plots/train{output_file}.png", bbox_inches="tight")
    return 0


if __name__ == '__main__':
    main()
