import random

from dataset import *
from utils.eval import *
from models.unireplknet import unireplknet_n
from models.de_unireplknet import de_unireplknet_n
import configparser
import logging


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def cal_anomaly_map_tensor(fs_list, ft_list, out_size=224, amap_mode='add'):
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='nearest')
        a_map_i = a_map[:, 0, :, :]
        a_map_list.append(a_map_i)

    map_a = torch.stack(a_map_list)
    if amap_mode == 'add':
        return map_a[0] + map_a[1] + map_a[2]
    else:
        return map_a[0] * map_a[1] * map_a[2]


def train(dataset):
    setup_seed(111)
    epochs = 130
    learning_rate = 5e-3
    batch_size = 16

    ckp_path = os.path.join(config['Outputs']['ckp_dir'], '{}_{}'.format(dataset, proj))
    device = 'cuda:{}'.format(config['System']['gpu_id']) if torch.cuda.is_available() else 'cpu'

    logging.info(device)

    train_data = AeBAD_SDataset(config['Datasets']['AeBAD_path'], 'AeBAD_S', "", split=DatasetSplit.TRAIN, )
    class_list = ['same', 'view', 'background', 'illumination']
    test_datas = []
    for _class in class_list:
        test_datas.append(AeBAD_SDataset(config['Datasets']['AeBAD_path'], 'AeBAD_S', _class, split=DatasetSplit.TEST))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloaders = []
    for j in range(len(test_datas)):
        test_dataloaders.append(torch.utils.data.DataLoader(test_datas[j], batch_size=1, shuffle=False, num_workers=4))

    encoder, bn = unireplknet_n()
    decoder = de_unireplknet_n()

    checkpoint = torch.load(config['Pretrain']['unireplknet_n'], map_location='cpu')
    if 'models' in checkpoint:
        checkpoint = checkpoint['models']
    encoder.load_state_dict(checkpoint, strict=False)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder.eval()

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        tqdm_obj = tqdm(range(len(train_dataloader)))
        for iteration, (img, label) in zip(tqdm_obj, train_dataloader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            # feature-poor datasets to compute global loss
            loss = loss_fucntion(inputs, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        ExpLR.step()
        logging.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
    torch.save({'bn': bn.state_dict(),
                'decoder': decoder.state_dict()}, ckp_path + '.pth')
    mean_auroc_sp = 0
    mean_aupro_px = 0
    for j in range(len(test_dataloaders)):
        _, auroc_sp, aupro_px = evaluate(encoder, bn, decoder, test_dataloaders[j], device, False,
                                         is_bool=False)
        mean_auroc_sp += auroc_sp
        mean_aupro_px += aupro_px
        logging.info("{}_auroc_sp:{}".format(class_list[j], auroc_sp))
        logging.info("{}_aupro_px:{}".format(class_list[j], aupro_px))
    mean_auroc_sp /= len(test_dataloaders)
    mean_aupro_px /= len(test_dataloaders)
    logging.info('mean_auroc_sp{:.3f}'.format(mean_auroc_sp))
    logging.info('mean_aupro_px{:.3f}'.format(mean_aupro_px))


if __name__ == '__main__':
    proj = "sep"
    config = configparser.ConfigParser()
    config.read('config.ini')
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler('{}/{}.log'.format(config['Outputs']['log_dir'], proj), mode='a'),
            logging.StreamHandler()
        ],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    train("AeBAD_S")
