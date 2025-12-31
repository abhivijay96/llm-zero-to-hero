from typing import List, Dict
import torch.nn.functional as F
import torch

class BigramTrainer:

    def _load_data(self):
        data = []
        with open('../data/names.txt', 'r') as data_file:
            for line in data_file:
                data.append(line.strip())
        self.names: List[str] = data[: 1]

    
    def _preprocess_data(self):
        self.vocab: List[str] = sorted(list(set('abcdefghijlkmnopqrstuvwxyz')))
        self.vocab = ['.'] + self.vocab

        self.ctoi: Dict[str, int] = {}

        for idx, c in enumerate(self.vocab):
            self.ctoi[c] = idx

        self.itoc: Dict[int, str] = {}
        for c in self.ctoi:
            self.itoc[self.ctoi[c]] = c

    def __init__(self):
        self._load_data()
        self._preprocess_data()
        self.reset_seed()
        self.W = torch.randn((len(self.vocab), len(self.vocab)), requires_grad=True, generator=self.g)
    
    def reset_seed(self):
        self.g = torch.Generator().manual_seed(2147483647)

    def fit(self):
        
        data = []
        labels = []

        for word in self.names:
            word = '.' + word + '.'
            for ch1, ch2 in zip(word, word[1:]):
                data.append(self.ctoi[ch1])
                labels.append(self.ctoi[ch2])
        
        # for d, l in zip(data, labels):
        #     print(f'{self.itoc[d]}, {self.itoc[l]}, {d}, {l}')
        
        data = torch.tensor(data)
        labels = torch.tensor(labels)
        print(f'data={data}, label={labels}')
        print(f'Shape of labels: {labels.shape}')

        data_enc = F.one_hot(data, num_classes=len(self.vocab)).float()

        for _ in range(1):
            
            logits = data_enc @ self.W
            # output of size C x 27
            counts = logits.exp()
            probabilities = counts / counts.sum(dim=1, keepdim=True)
            print(f'Shape of probabilities: {probabilities.shape}')

            loss = -probabilities[torch.arange(data.nelement()), labels].log().mean()

            print(f'Loss={loss.item()}')

            self.W.grad = None
            loss.backward()
            self.W.data += -50 * self.W.grad


        # nll = torch.zeros(5)

        # for i in range(labels.nelement()):
        #     x = data[i]
        #     y = labels[i]

        #     probs = probabilities[i]
        #     predicted_prob = probs[y.item()]

        #     logprob = predicted_prob.log()
        #     nll[i] = -logprob

        #     print(f'Probability of {self.itoc[x.item()]} predicting {self.itoc[y.item()]} = {logprob.item()}')

        # print(f'Manually calculated loss={nll.mean().item()}')



    def generate(self):
        start = 0
        out = []

        while True:

            start_enc = F.one_hot(torch.tensor([start]), num_classes=len(self.vocab)).float()
            logits = start_enc @ self.W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)

            # IMPORTANT: Did not use the loss here, used the previous activation's output
            start = torch.multinomial(probs, num_samples=1, replacement=True, generator=self.g).item()

            if start == 0:
                break

            out.append(self.itoc[start])

        return ''.join(out)
        
        
    


trainer = BigramTrainer()
trainer.fit()
trainer.reset_seed()
print(trainer.generate())
print(trainer.generate())
print(trainer.generate())