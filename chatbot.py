import torch 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
class Chatbot:
    def __init__(self, max_length, temperature, no_repeat_ngram_size , num_beams):

        #initializing variable for the generationg function
        self.max_length = max_length
        self.temperature =temperature
        self.num_beams =num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size

        #initializing model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
    

    #function to take inputs and handling, return the attention mask and the input ids
    def input_handling(self, inputs):

        self.input_ids = self.tokenizer.encode(inputs, return_tensors ="pt")
        self.attention_mask = torch.ones_like(self.input_ids)
        return self.input_ids, self.attention_mask

    #function to respond, return a print of the respond generated for the model
    def respond(self):
        #generating respond
        self.outputs = self.model.generate(
        self.input_ids,
        max_length=self.max_length,
        num_beams=self.num_beams,
        temperature=self.temperature,
        no_repeat_ngram_size=self.no_repeat_ngram_size,
        pad_token_id=self.tokenizer.eos_token_id,
        bos_token_id=self.tokenizer.bos_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
        do_sample=True,
        attention_mask = self.attention_mask
        )

        #responding
        self.response = self.tokenizer.decode(self.outputs[0], skip_special_tokens = True)
        self.response = self.response[len(self.input_ids):].strip()
        return str(self.response)
