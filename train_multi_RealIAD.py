import configparser
import random

from torch.utils.data import DataLoader, ConcatDataset

from dataset import *
from models.de_unireplknet import de_unireplknet_n, de_unireplknet_s, de_unireplknet_t
from models.unireplknet import unireplknet_n, unireplknet_s, unireplknet_t
from utils.eval import *
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


def loss_adc(a, b, p_hard, p_lim, num_lim, map_size):
    anomaly_map = cal_anomaly_map_tensor(a, b, map_size)
    _std = torch.std(anomaly_map)
    d_hard = torch.quantile(anomaly_map, q=p_hard) - _std * _std
    a_map = anomaly_map * anomaly_map
    num = torch.sum(a_map >= d_hard)
    if num < num_lim:
        d_hard = torch.quantile(a_map, q=p_lim) - _std * _std
        # num = torch.sum(a_map >= d_hard)
    loss = (a_map[a_map >= d_hard]).mean()
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
    seed = 111
    setup_seed(seed)
    epochs = 1
    learning_rate = 5e-3
    batch_size = 16
    image_size = 256
    p_hard = 0.9999
    p_lim = 0.9995
    num_lim = image_size * image_size * batch_size * (1 - p_lim)
    logging.info("num_lim={}".format(num_lim))

    ckp_path = os.path.join(config['Outputs']['ckp_dir'], '{}_{}'.format(dataset, proj))
    device = 'cuda:{}'.format(config['System']['gpu_id']) if torch.cuda.is_available() else 'cpu'

    logging.info(device)
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    class_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                  'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                  'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                  'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                  'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    train_data_list = []
    test_data_list = []
    for i, item in enumerate(class_list):
        train_data = RealIADDataset(root=config['Datasets']['RealIAD_path'], item=item, transform=data_transform,
                                    gt_transform=None,
                                    phase="train")
        test_data = RealIADDataset(root=config['Datasets']['RealIAD_path'], item=item, transform=data_transform,
                                   gt_transform=gt_transform,
                                   phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)
    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    test_dataloader_list = [
        torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        for test_data in test_data_list]

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
        for iteration, (img, _) in zip(tqdm_obj, train_dataloader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            # feature-rich datasets to compute adc loss
            loss = loss_adc(inputs, outputs, p_hard, p_lim, num_lim, image_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        ExpLR.step()
        logging.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
    torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path + '.pth')
    results_list = []
    for j in range(len(test_dataloader_list)):
        results = evaluation_batch(encoder, bn, decoder,
                                   test_dataloader_list[j],
                                   device,
                                   max_ratio=0.01)
        results_list.append(results)
    logging.info(results_list)
    mean_auroc_sp = np.mean([result[0] for result in results_list])
    mean_ap_sp = np.mean([result[1] for result in results_list])
    mean_f1_sp = np.mean([result[2] for result in results_list])
    mean_auroc_px = np.mean([result[3] for result in results_list])
    mean_ap_px = np.mean([result[4] for result in results_list])
    mean_f1_px = np.mean([result[5] for result in results_list])
    mean_aupro_px = np.mean([result[6] for result in results_list])
    logging.info(
        "Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}"
        .format(mean_auroc_sp, mean_ap_sp, mean_f1_sp, mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))
    logging.info("done.")


if __name__ == '__main__':
    proj = "uni"
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
    train("RealIAD")
