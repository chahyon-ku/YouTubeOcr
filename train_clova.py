import time

import tqdm
import korean_util as ku
import torch
import data_clova


def train_one_epoch(model, loss_fn, optim, data_loader, DEVICE):
    model.train()
    data_loader_tqdm = tqdm.tqdm(data_loader)
    batch_i = 0
    start = time.time()
    for imgs, inits, medials, finals, font_i in data_loader_tqdm:
        imgs = imgs.to(DEVICE)
        inits = inits.to(DEVICE)
        medials = medials.to(DEVICE)
        finals = finals.to(DEVICE)
        data_time = time.time() - start

        logits = model(imgs)
        pred_inits = logits[:, :len(ku.CHAR_INITIALS)]
        pred_medials = logits[:, len(ku.CHAR_INITIALS):len(ku.CHAR_INITIALS)+len(ku.CHAR_MEDIALS)]
        pred_finals = logits[:, len(ku.CHAR_INITIALS)+len(ku.CHAR_MEDIALS):]
        loss_inits = loss_fn(pred_inits, inits)
        loss_medials = loss_fn(pred_medials, medials)
        loss_finals = loss_fn(pred_finals, finals)

        model.zero_grad()
        loss_inits.backward(retain_graph=True)
        loss_medials.backward(retain_graph=True)
        loss_finals.backward()
        optim.step()

        batch_time = time.time() - start
        start = time.time()

        loss_init = torch.sum(loss_inits).item()
        loss_medial = torch.sum(loss_inits).item()
        loss_final = torch.sum(loss_inits).item()
        accuracy_init = torch.sum(torch.argmax(pred_inits, dim=-1) == inits).cpu().item() / logits.shape[0]
        accuracy_medial = torch.sum(torch.argmax(pred_medials, dim=-1) == medials).cpu().item() / logits.shape[0]
        accuracy_final = torch.sum(torch.argmax(pred_finals, dim=-1) == finals).cpu().item() / logits.shape[0]
        data_loader_tqdm.set_postfix_str('data_time: {:.3g}, batch_time: {:.3g}, acc: {:.3g} {:.3f} {:.3f}'.format(data_time, batch_time, accuracy_init, accuracy_medial, accuracy_final))
        if batch_i % 10 == 0:
            torch.save(model.state_dict(), 'models/'+str(batch_i)+'.pth')
        batch_i += 1

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1024

    # data.generate_data()
    dataset = data_clova.CharDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None, num_workers=2,
                                              pin_memory=True)

    model = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 3, 1, 'same'),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(16, 16, 3, 1, 'same'),
                                torch.nn.ReLU(),
                                torch.nn.Flatten(),
                                torch.nn.Linear(16 * 32 * 32, 256),
                                torch.nn.ReLU(),
                                torch.nn.Linear(256,
                                                len(ku.CHAR_INITIALS) + len(ku.CHAR_MEDIALS) + len(ku.CHAR_FINALS) + 1))
    model = model.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        print(epoch)
        train_one_epoch(model, loss_fn, optim, data_loader, DEVICE)