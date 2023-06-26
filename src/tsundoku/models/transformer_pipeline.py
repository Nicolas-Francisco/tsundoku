from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from textwrap import wrap

from tsundoku.models.transformer import BETOTweeterClassifier, BETOTokenizer, BETOModel
from tsundoku.models.dataset_class import TsundokuUsersDataset


EPOCHS = 3
NCLASSES = 17

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BETOTweeterClassifier(NCLASSES)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_fn = nn.CrossEntropyLoss().to(device)


def data_loader(df, tokenizer, label_encoder, max_len, batch_size):
    dataset = TsundokuUsersDataset(
        descriptions=df.description.to_numpy(),
        locations=df.location.to_numpy(),
        names=df.name.to_numpy(),
        screen_names=df.screen_name.to_numpy(),
        urls=df.url.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_len=max_len,
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=2)


def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    index = 1
    for batch in data_loader:
        print(f"current batch: n°{index}")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        index += 1
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def classifyDescription(text, tokenizer, MAX_LEN=200):
    encoding_description = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        truncation=True,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding_description["input_ids"].to(device)
    attention_mask = encoding_description["attention_mask"].to(device)
    output = model(input_ids, attention_mask)
    _, labeled_class = torch.max(output, dim=1)
    print("\n".join(wrap(text)))
    print(str(labeled_class))
    return


def execute_transformer_pipeline(
    train_data_loader, df_train, validation_data_loader, df_validation, output_dir
):
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(EPOCHS):
        print("Epoch {} de {}".format(epoch + 1, EPOCHS))
        print("------------------")
        train_acc, train_loss = train_model(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train),
        )
        print("Entrenamiento: Loss: {}, accuracy: {}".format(train_loss, train_acc))
        validation_acc, validation_loss = eval_model(
            model, validation_data_loader, loss_fn, device, len(df_validation)
        )
        print(
            "Validación: Loss: {}, accuracy: {}".format(validation_loss, validation_acc)
        )
        print("")
    model.beto.save_pretrained(output_dir)
    BETOTokenizer.save_pretrained(output_dir)
    return
