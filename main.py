from datasets import load_dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from torch.utils.data import DataLoader, Subset
import evaluate as evaluate
from tqdm import tqdm
import torch
from transformers import get_scheduler
from transformers import DataCollatorWithPadding
import time

def evaluate_model(model, dataloader, device):

    
    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    # iterate over the dataloader
    for batch in dataloader:

        labels, input_ids, masks = batch['labels'], batch['input_ids'], batch['attention_mask']      
        labels=labels.to(device)
        input_ids=input_ids.to(device)
        masks=masks.to(device)

        # forward pass
        # name the output as `output`
        output = model(input_ids, attention_mask=masks)
        predictions = output['logits']


        predictions = torch.argmax(predictions, dim=-1)
        ## Flatten predictions and labels to match evaluation's format
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()
        dev_accuracy.add_batch(predictions=predictions_flat, references=labels_flat)

    # compute and return metrics
    return dev_accuracy.compute()

def train(mymodel, model_name, save_path, train_dataloader, validation_dataloader, lr, num_epochs, device):
    mymodel.to(device)
    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    loss = torch.nn.CrossEntropyLoss().to(device)
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print("Epoch "+str(epoch + 1)+" training:")
        for i, batch in tqdm(enumerate(train_dataloader)):
            ## Get input_ids, attention_mask, labels
            labels, input_ids, masks = batch['labels'], batch['input_ids'], batch['attention_mask']
            ## Send to GPU
            labels=labels.to(device)
            input_ids=input_ids.to(device)
            masks=masks.to(device)
            ## Forward pass
            output = mymodel(input_ids, attention_mask=masks)
            predictions = output['logits']
            ## Cross-entropy loss
            # print('pred',predictions.view(-1, mymodel.num_labels))
            # print('lab',labels.view(-1))
            centr = loss(predictions.view(-1, mymodel.num_labels), labels.view(-1))
            #print('loss',centr)
            ## Backward pass
            centr.backward()
            ## Update Model
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ## Argmax get real predictions
            predictions = torch.argmax(predictions, dim=-1)
            ## Flatten predictions and labels to match evaluation's format
            predictions_flat = predictions.flatten()
            labels_flat = labels.flatten()
            # print('pred',predictions_flat)
            # print('lab',labels_flat)
            # print("Shape of predictions:", predictions_flat.shape, predictions_flat.dtype)  
            # print("Shape of labels:", labels_flat.shape, labels_flat.dtype)  
            ## Update metrics
            train_accuracy.add_batch(predictions=predictions_flat, references=labels_flat)
        # print evaluation metrics
        print(" ===> Epoch "+ str(epoch + 1))
        train_acc = train_accuracy.compute()
        # print(" - Average training metrics: accuracy={train_acc}")
        print(" - Average training metrics: accuracy="+str(train_acc))
        train_acc_list.append(train_acc['accuracy'])

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        
        print(f" - Average validation metrics: accuracy="+str(val_accuracy))
        dev_acc_list.append(val_accuracy['accuracy'])
        
        epoch_list.append(epoch)
        
        # test_accuracy = evaluate_model(mymodel, test_dataloader, device)
        # print(f" - Average test metrics: accuracy={test_accuracy}")

        epoch_end_time = time.time()
        #print("Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")
        print("Epoch "+str(epoch + 1)+" took "+str(epoch_end_time - epoch_start_time)+" seconds")
    ## Save model and push to HF hub
    torch.save(mymodel.state_dict(), save_path)
    mymodel.push_to_hub(model_name)
    tokenizer.push_to_hub(model_name)
    return train_acc_list, dev_acc_list, epoch_list

def tokenization(example):
    return tokenizer(example["sentence"], truncation=True)
def preprocess(dataset, tokenizer):
    '''
    We only do this once and then we store the processed dataset
    One dataset at a time
    '''
    preprocess = Preprocess(dataset=dataset, tokenizer=tokenizer)
    tokenized_dataset = preprocess.preprocess()
    return tokenized_dataset

if __name__=="__main__":
    ## Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "; using gpu:", torch.cuda.get_device_name())

    # Dataset
    
    dataset = load_dataset("stanfordnlp/sst2")
    train_data, test_data = load_dataset("stanfordnlp/sst2", split =['train', 'validation'])
    train_val_splitdata = train_data.train_test_split(test_size=0.1)
    train_data = train_val_splitdata['train']
    validation_data = train_val_splitdata['test']

    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512, return_tensors="pt")

    ## Small dataset for testing
    splitted_train_data = train_data.train_test_split(test_size=0.005)
    small_data = splitted_train_data['test']

    # Data size using
    data_size = "small" #full or small
    if data_size == "small":
        train_data = small_data
    
    # Tokenize the entire dataset
    train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
    validation_data = validation_data.map(tokenization, batched = True, batch_size = len(validation_data))
    test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))


    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    validation_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32, collate_fn=data_collator)
    validation_dataloader = DataLoader(validation_data, shuffle=True, batch_size=32, collate_fn=data_collator)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=32, collate_fn=data_collator)


    # Set training parameters
    training_type="LORA" #LORA, full
    lr=0.00001
    num_epochs=20
    model_name="adv-ssm-hw1-"+training_type+"Para-"+data_size+"Data-"+str(round(time.time()))
    save_path='/home/cs601-pwang71/adv-ssm-hw1/saved_models/'+model_name+'.pt'
    # Log Training Info
    print("Training info:")
    print("Training type:", training_type, "Learning rate:", lr, "Num Epochs:", num_epochs, "Data Size:", data_size)
    print("Model Name:", model_name, "Save Path:", save_path)

    # PeFT
    if training_type=="LORA":
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    # Train

    train_acc_list, dev_acc_list, epoch_list = train(mymodel=model, model_name=model_name, save_path=save_path, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, lr=lr, num_epochs=num_epochs, device=device)#0.00001


    