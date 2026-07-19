import torch

class CFDNATokenizer:

    def __init__(self):
        #total 12 vocab size
        self.vocab ={
            "<pad>":0,
            "<cfdna>":1,
            "</cfdna>":2,

            "A":3,
            "C":4,
            "G":5,
            "T":6,

            "<m>":7,
            "<um>":8,

            "<mask>":9,
            "<cls>":10,
            "<sep>":11
                        
        }

        #inverse vocab so from <string , int> to <int ,string> 
        self.inverse_vocab = {
            v:k for k,v in self.vocab.items()
        }

        self.vocab_size = len(self.vocab)

    

    def encode(self, tokens): 
        ids=[] 
        #will use collate for padding by batch max later 
        #max_raw_length= max_length -2 # 2 foe cls and sep 

        for token in tokens: # each fragment in fragments list have one token array of strings.
            if token not in self.vocab: #it will look if the token is NOT in our vocab  
                raise ValueError(f"Unknown token: {token}")
            
            ids.append(self.vocab[token])#fetch the key for that token and append

        # if len(ids) > max_raw_length: ## slicing the array from the start to max_raw_length
        #     ids=ids[:max_raw_length]


         #add cls in from , <cls> stands for "Classification" which we add at the start.
        ids = [
            self.vocab["<cls>"] 
        ]+ids
        
        #add <sep> at end ("Separation" (or end-of-sequence))
        ids.append(self.vocab["<sep>"])
        '''        ids = [ <cls>,  <cfdna>,  G,  G,  A,  ...,  </cfdna>,  <sep> ]
                             │        │                            │         │
                         (Front)      └─────────── Genomic ────────┘       (Back)
                        Added first                  Tokens               Appended last
        '''
       

        # if len(ids) < max_length:
        #     ids += [ self.vocab["<pad>"] ] * (max_length -  len(ids))  #[0] * 3 -> results in [0, 0, 0] (List duplication) and then attach at the end of orignal list
    
        return torch.tensor(ids, dtype=torch.long)
    



    def decode(self, ids):
        return[self.inverse_vocab[i] for i in ids if i in self.inverse_vocab]



# tokenizer=CFDNATokenizer()


# fragment=[
# "<cfdna>",
# "G",
# "G",
# "A",
# "<m>",
# "C",
# "<um>",
# "T",
# "</cfdna>"
# ]


# encoded = tokenizer.encode(fragment)


# print(encoded)

# print(tokenizer.decode(encoded.tolist()))