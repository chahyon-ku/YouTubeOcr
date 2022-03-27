import time

import tqdm

import char_class_model
import korean_util as ku
import torch
import data_clova
import tensorboardX


def train_one_epoch(model, loss_fn, optim, data_loader, DEVICE, epoch_i):
    writer = tensorboardX.SummaryWriter('log')

    model.train()
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
        loss_inis = loss_fn(logit_inis, inis)
        loss_meds = loss_fn(logit_meds, meds)
        loss_fins = loss_fn(logit_fins, fins)

        model.zero_grad()
        loss_inis.backward(retain_graph=True)
        loss_meds.backward(retain_graph=True)
        loss_fins.backward()
        optim.step()

        loss_ini = torch.sum(loss_inis).item()
        loss_med = torch.sum(loss_inis).item()
        loss_fin = torch.sum(loss_inis).item()
        pred_inis = torch.argmax(logit_inis, dim=-1)
        pred_meds = torch.argmax(logit_meds, dim=-1)
        pred_fins = torch.argmax(logit_fins, dim=-1)
        accuracy_ini = torch.sum(pred_inis == inis).cpu().item() / logits.shape[0]
        accuracy_med = torch.sum(pred_meds == meds).cpu().item() / logits.shape[0]
        accuracy_fin = torch.sum(pred_fins == fins).cpu().item() / logits.shape[0]
        accuracy_all = torch.sum((pred_inis == inis) & (pred_meds == meds) & (pred_fins == fins)).cpu().item() / logits.shape[0]
        writer.add_scalar('loss_ini', loss_ini, batch_i + epoch_i * len(data_loader))
        writer.add_scalar('loss_med', loss_med, batch_i + epoch_i * len(data_loader))
        writer.add_scalar('loss_fin', loss_fin, batch_i + epoch_i * len(data_loader))
        writer.add_scalar('accuracy_ini', accuracy_ini, batch_i + epoch_i * len(data_loader))
        writer.add_scalar('accuracy_med', accuracy_med, batch_i + epoch_i * len(data_loader))
        writer.add_scalar('accuracy_fin', accuracy_fin, batch_i + epoch_i * len(data_loader))
        writer.add_scalar('accuracy_all', accuracy_fin, batch_i + epoch_i * len(data_loader))

        batch_time = time.time() - start
        start = time.time()

        data_loader_tqdm.set_postfix_str('data_time: {:.3g}, batch_time: {:.3g}, acc: {:.3g} {:.3f} {:.3f} {:.3f}'.format(data_time, batch_time, accuracy_ini, accuracy_med, accuracy_fin, accuracy_all))
        if batch_i % 10 == 0:
            torch.save(model.state_dict(), 'models/' + str(batch_i) + '.pth')
        batch_i += 1


if __name__ == '__main__':
    DEVICE = torch.device('cuda')
    BATCH_SIZE = 1024

    # data.generate_data()
    dataset = data_clova.CharColorDataset('data/clova-32.h5')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None, num_workers=1,
                                              pin_memory=True)

    model = char_class_model.CharClassModel(3, 32, len(ku.CHAR_INITIALS) + len(ku.CHAR_MEDIALS) + len(ku.CHAR_FINALS) + 1, 32)
    print('n_parameters:', sum(p.numel() for p in model.parameters()))
    model = model.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch_i in range(1):
        train_one_epoch(model, loss_fn, optim, data_loader, DEVICE, epoch_i)
