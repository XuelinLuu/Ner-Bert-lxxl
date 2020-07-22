import tqdm
import torch
import torch.nn as nn

def loss_fn(outputs, labels):
    return nn.BCEWithLogitsLoss()(outputs, labels)

def train_fn(model, device, train_dataloader, optimizer, lr_scheduler=None):
    model.train()
    tk = tqdm.tqdm(train_dataloader, desc="Train Iter")
    train_loss = 0
    for idx, data in enumerate(tk):
        input_ids, attention_mask, token_type_ids, labels = data[0], data[1], data[2], data[3]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        train_loss += loss.item()
        avg_loss = train_loss / (idx+1)
        tk.set_postfix(loss=loss.item(), avg_loss=avg_loss)

def eval_fn(model, device, eval_dataloader, eval_dir):
    model.eval()
    tk = tqdm.tqdm(eval_dataloader, desc="Train Iter")
    eval_loss = 0
    eval_acc = 0
    acc_step = 0
    with torch.no_grad():
        for idx, data in enumerate(tk):
            input_ids, attention_mask, token_type_ids, labels = data[0], data[1], data[2], data[3]
            real_len = data[4]

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(outputs, labels)

            pred = torch.argmax(outputs, dim=-1)
            pred = pred.detach().cpu().numpy()
            labels = torch.argmax(labels, dim=-1)
            labels = labels.detach().cpu().numpy()

            for batch_idx in range(len(labels)):
                real_leng = real_len.numpy()[batch_idx]
                for seq_idx in range(real_leng):
                    acc_step += 1
                    if pred[batch_idx][seq_idx] == labels[batch_idx][seq_idx]:
                        eval_acc += 1



            eval_loss += loss.item()
            avg_loss = eval_loss / (idx + 1)

            avg_acc = eval_acc / acc_step
            tk.set_postfix(acc=eval_acc, avg_acc=avg_acc, loss=loss.item(), avg_loss=avg_loss)
            with open(eval_dir, "a", encoding="utf-8") as f:
                f.write(f"idx={idx}, acc={eval_acc}, avg_acc={avg_acc}, loss={loss.item()} avg_loss={avg_loss}\n")


