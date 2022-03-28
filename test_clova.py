import cv2
import numpy
import torch

import char_class_model
import korean_util as ku
import data_clova
import tensorboardX
import tqdm
import time
import matplotlib.pyplot as plt


def test_one_epoch(model, loss_fn, data_loader, DEVICE):
    model.eval()
    data_loader_tqdm = tqdm.tqdm(data_loader)
    batch_i = 0
    start = time.time()
    for imgs, inis, meds, fins, font_i in data_loader_tqdm:
        imgs = imgs.to(DEVICE)
        inis = inis.to(DEVICE)
        meds = meds.to(DEVICE)
        fins = fins.to(DEVICE)
        data_time = time.time() - start

        logits = model(imgs).squeeze()
        logit_inis = logits[:, :len(ku.CHAR_INITIALS)]
        logit_meds = logits[:, len(ku.CHAR_INITIALS):len(ku.CHAR_INITIALS)+len(ku.CHAR_MEDIALS)]
        logit_fins = logits[:, len(ku.CHAR_INITIALS)+len(ku.CHAR_MEDIALS):]
        pred_inis = torch.argmax(logit_inis, dim=-1)
        pred_meds = torch.argmax(logit_meds, dim=-1)
        pred_fins = torch.argmax(logit_fins, dim=-1)

        mask_inis = pred_inis != inis
        mask_meds = pred_meds != meds
        mask_fins = pred_fins != fins
        mask_all = mask_inis | mask_meds | mask_fins

        for img_i in range(len(imgs)):
            if mask_all[img_i]:
                pred_ini = ku.CHAR_INITIALS[pred_inis[img_i].cpu().item()]
                pred_med = ku.CHAR_MEDIALS[pred_meds[img_i].cpu().item()]
                pred_fin = ku.CHAR_FINALS[pred_fins[img_i].cpu().item() - 1] if pred_fins[img_i].cpu().item() > 0 else ''
                ini = ku.CHAR_INITIALS[inis[img_i].cpu().item()]
                med = ku.CHAR_MEDIALS[meds[img_i].cpu().item()]
                fin = ku.CHAR_FINALS[fins[img_i].cpu().item() - 1] if fins[img_i].cpu().item() > 0 else ''
                print('{:d}, {} {}, {} {}, {} {}'.format(img_i, pred_ini, ini, pred_med, med, pred_fin, fin))
                img = numpy.array(imgs[img_i].cpu() * 255, dtype=numpy.uint8)
                img = numpy.swapaxes(img, 0, 1)
                img = numpy.swapaxes(img, 1, 2)
                cv2.imshow('img', img)
                cv2.waitKey()


        loss_inis = loss_fn(logit_inis, inis)
        loss_meds = loss_fn(logit_meds, meds)
        loss_fins = loss_fn(logit_fins, fins)

        batch_time = time.time() - start
        start = time.time()

        loss_ini = torch.sum(loss_inis).item()
        loss_med = torch.sum(loss_inis).item()
        loss_fin = torch.sum(loss_inis).item()
        accuracy_ini = torch.sum(pred_inis == inis).cpu().item() / logits.shape[0]
        accuracy_med = torch.sum(pred_meds == meds).cpu().item() / logits.shape[0]
        accuracy_fin = torch.sum(pred_fins == fins).cpu().item() / logits.shape[0]
        accuracy_all = torch.sum((pred_inis == inis) & (pred_meds == meds) & (pred_fins == fins)).cpu().item() / logits.shape[0]
        data_loader_tqdm.set_postfix_str('data_time: {:.3g}, batch_time: {:.3g}, acc: {:.3g} {:.3f} {:.3f} {:.3f}'.format(data_time, batch_time, accuracy_ini, accuracy_med, accuracy_fin, accuracy_all))
        batch_i += 1


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    dataset = data_clova.CharColorDataset('data/nanum-32.h5')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None, num_workers=1,
                                              pin_memory=True)

    model = char_class_model.CharClassModel(3, 32, len(ku.CHAR_INITIALS) + len(ku.CHAR_MEDIALS) + len(ku.CHAR_FINALS) + 1, 32)
    model.load_state_dict(torch.load('models/nanum-c-32-3.pth'))
    model = model.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())
    #writer = tensorboardX.SummaryWriter('../log/clova/'+str(int(time.time())))

    for epoch in range(1):
        test_one_epoch(model, loss_fn, data_loader, DEVICE)