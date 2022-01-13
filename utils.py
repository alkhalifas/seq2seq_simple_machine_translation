import spacy
import torch
from torchtext.data.metrics import bleu_score
import sys

spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')


def translate_sentence(model, sentence, german, english, device, max_length=50):
    spacy_ger = spacy.load("de_core_news_sm")
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    """
    Calculates BLEU score
    :param data: Input data
    :param model: Model of interest
    :param german: German text
    :param english: English text
    :param device: device of interest
    :return: BLEU score
    """
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Saves the model checkpoint
    :param state: State of the mode
    :param filename: Path to save
    :return:
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """
    Loads the model checkpoint
    :param checkpoint:
    :param model:
    :param optimizer:
    :return:
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def tokenizer_ger(text):
    """
    Tokenize a string into a list of words
    :param text: A sentence in the form of a string
    :return: A list of words from that sentence
    """
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    """
    Tokenize a string into a list of words
    :param text: A sentence in the form of a string
    :return: A list of words from that sentence
    """
    return [tok.text for tok in spacy_eng.tokenizer(text)]
