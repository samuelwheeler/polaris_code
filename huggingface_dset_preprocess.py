from transformers import GPT2Tokenizer
from itertools import chain
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
from config import config

# from https://discuss.huggingface.co/t/make-text-data-continuous-from-datasetdict/17812/2

def pre_pro(seq):
    seq['text'] = [s.split('\n\nSee also')[0] + '<|endoftext|>' for s in seq['text']]
    return seq


def tokenize_map(examples, seq_len):
    seq_length = seq_len+1
    examples = pre_pro(examples)
    examples = tokenizer(examples["text"])
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= seq_length:
        total_length = (total_length // seq_length) * seq_length
    result = {
        k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
        for k, t in concatenated_examples.items()
    }
    result = {k: v for k, v in result.items() if len(v[0]) == seq_length}
    return result
tokenize = lambda x: tokenize_map(x, config.train_seq_len)

# if __name__ == "__main__":
        
#     ds = load_dataset("c4", "en", split = 'train', streaming = True)
#     ds = ds.shuffle(seed = 42, buffer_size = 10_000)
#     ds = ds.map(tokenize, batched = True, remove_columns = ['text', 'timestamp', 'url', 'attention_mask'])

#     x = list(ds.take(4))
    
#     import pdb; pdb.set_trace()




