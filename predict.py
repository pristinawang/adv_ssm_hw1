from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Subset
import evaluate as evaluate
from tqdm import tqdm
import torch
from transformers import get_scheduler
from transformers import DataCollatorWithPadding
import time

def predict_model(model, dataloader, device):
    '''
    return test set accuracy
    '''
    model.to(device)
    # load metrics
    test_accuracy = evaluate.load('accuracy')

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
        test_accuracy.add_batch(predictions=predictions_flat, references=labels_flat)

    # compute and return metrics
    return test_accuracy.compute()

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

#     model = RobertaForSequenceClassification.from_pretrained('roberta-base')
#     tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512, return_tensors="pt")

#     # Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_id = "pristinawang/adv-ssm-hw1-bitfitPara-smallData-1726341791"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)


    ## Small dataset for testing
    # splitted_train_data = train_data.train_test_split(test_size=0.005)
    # small_data = splitted_train_data['test']

    # # Data size using
    # data_size = "full"
    # if data_size == "small":
    #     train_data = small_data
    
    # Tokenize the entire dataset
    # train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
    # validation_data = validation_data.map(tokenization, batched = True, batch_size = len(validation_data))
    test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))


    # train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # validation_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32, collate_fn=data_collator)
    # validation_dataloader = DataLoader(validation_data, shuffle=True, batch_size=32, collate_fn=data_collator)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=32, collate_fn=data_collator)


    # Set training parameters
    # training_type="full"
    # lr=0.00001
    # num_epochs=20
    # model_name="adv-ssm-hw1-"+training_type+"Para-"+data_size+"Data-"+str(round(time.time()))
    # save_path='/home/cs601-pwang71/adv-ssm-hw1/saved_models/'+model_name+'.pt'
    # # Log Training Info
    # print("Training info:")
    # print("Training type:", training_type, "Learning rate:", lr, "Num Epochs:", num_epochs, "Data Size:", data_size)
    # print("Model Name:", model_name, "Save Path:", save_path)

    # Train

    #train_acc_list, dev_acc_list, epoch_list = train(mymodel=model, model_name=model_name, save_path=save_path, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, lr=lr, num_epochs=num_epochs, device=device)#0.00001
    test_accuracy=predict_model(model=model, dataloader=test_dataloader, device=device)
    print(test_accuracy)

    