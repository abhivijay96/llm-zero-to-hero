from typing import List, Dict
import torch
import logging
import torch.nn.functional as F


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPTrainer:

    def extract_data(self) -> List[str]:
        data = []
        with open('../data/names.txt', 'r') as data_file:
            for line in data_file:
                line = line.strip()
                data.append(line)
        return data

    def reset_seed(self) -> torch.Generator:
        return torch.Generator().manual_seed(2147483647)


    def __init__(self, context_window=3):
        
        self.data = self.extract_data()
        self.g = self.reset_seed()

        self.vocab: List[str] = list('abcdefghijklmnopqrstuvwxyz')

        self.ctoi: Dict[str, int] = {c: i + 1 for i, c in enumerate(self.vocab)}
        self.ctoi['.'] = 0

        self.iotc: Dict[int, str] = {self.ctoi[c]: c for c in self.ctoi}

        self.embedding_dim = 8

        self.context_size = context_window

        self.embedding_table = torch.randn((27, self.embedding_dim), generator=self.g, requires_grad=True)
        self.W1 = torch.randn((self.embedding_dim * self.context_size, 256), generator=self.g, requires_grad=True)
        self.b1 = torch.randn((256), generator=self.g, requires_grad=True)
        self.W2 = torch.randn((256, 27), generator=self.g, requires_grad=True)
        self.b2 = torch.randn((27), generator=self.g, requires_grad=True)

        self.params = [self.embedding_table, self.W1, self.b1, self.W2, self.b2]


    def get_param_count(self):
        total = 0
        for param in self.params:
            total += param.nelement()
        return total


    def load_dataset(self, words):
        xs = []
        ys = []

        for word in words:
            word = word + '.'

            context = [0] * self.context_size

            for c in word:
                xs.append(context)
                ys.append(self.ctoi[c])

                # logger.info(f'{"".join([self.iotc[i] for i in context])} ---> {c}')

                context = context[1: ] + [self.ctoi[c]]
        
        return torch.Tensor(xs).long(), torch.Tensor(ys).long()

    def cmp(self, dt: torch.Tensor, t: torch.Tensor, name: str):
        ex = torch.all(dt == t.grad)
        close = torch.allclose(dt, t.grad)
        max_diff = (dt - t.grad).abs().max().item()
        logger.info(f'Exact match={str(ex):15s} | Close match={str(close):15s} | Max diff={str(f"{max_diff:.4f}") :15s} | {name}')
        return ex, close


    def fit(self):

        xs, ys = self.load_dataset(self.data)
        

        logger.info(f'xs={xs.shape}')
        logger.info(f'ys={ys.shape}')
        logger.info(f'Parameter count={self.get_param_count()}')

        for step in range(3000):
            ix = torch.randint(0, xs.shape[0], (64, ))

            embeddings = self.embedding_table[xs[ix]]
            hidden_pact = embeddings.view(-1, self.embedding_dim * self.context_size) @ self.W1 + self.b1
            hidden = torch.tanh(hidden_pact)
            logits = hidden @ self.W2 + self.b2
            exponents = logits.exp()
            exponents_sum = exponents.sum(dim=1, keepdim=True)
            exponents_inv = exponents_sum**-1
            probs = exponents * exponents_inv
            logprobs = -probs[torch.arange(probs.shape[0]), ys[ix]].log()
            loss = logprobs.mean()

            logger.info(f'loss={loss.item()}')

            for param in self.params:
                param.grad = None

            embeddings.retain_grad()
            hidden_pact.retain_grad()
            hidden.retain_grad()
            logits.retain_grad()
            exponents.retain_grad()
            exponents_sum.retain_grad()
            exponents_inv.retain_grad()
            probs.retain_grad()
            logprobs.retain_grad()
            loss.retain_grad()

            loss.backward()

            lr = 0.1
            if step > 10000:
                lr = 0.05
            
            for param in self.params:
                param.data += -lr * param.grad
            

            # TO test learning rate, use the following
            # torch.linspace to sample learning rate exponents,
            # Using these exponents, sample learning rates

        # Manually compute gradients
        dloss = torch.ones_like(loss)
        self.cmp(dloss, loss, 'dloss')
        dlogprobs = dloss * (1 / logprobs.shape[0])
        self.cmp(dlogprobs, logprobs, 'dlogprobs')
        dprobs = torch.zeros_like(probs)
        dprobs[torch.arange(probs.shape[0]), ys[ix]] = (-1 / probs[torch.arange(probs.shape[0]), ys[ix]]) * dlogprobs
        self.cmp(dprobs, probs, 'dprobs')
        dexponents_inv = (exponents * dprobs).sum(dim=1, keepdim=True)
        self.cmp(dexponents_inv, exponents_inv, 'dexponents_inv')
        dexponents = dprobs * exponents_inv # this d(probs)/d(exponents)
        dexponents_sum = (-1 / exponents_sum**2) * dexponents_inv
        self.cmp(dexponents_sum, exponents_sum, 'dexponents_sum')

        dexponents += torch.ones_like(exponents) * dexponents_sum
        self.cmp(dexponents, exponents, 'dexponents')

        dlogits = dexponents * exponents
        self.cmp(dlogits, logits, 'dlogits')

        dhidden = dlogits @ self.W2.T
        self.cmp(dhidden, hidden, 'dhidden')

        dhidden_pact = dhidden * (1 - hidden**2)
        self.cmp(dhidden_pact, hidden_pact, 'dhidden_pact')

        # dembeddings = dhidden_pact.view(-1, self.embedding_dim * self.context_size) @ self.W1.T
        # self.cmp(dembeddings, embeddings, 'dembeddings')


    @torch.no_grad()
    def generate(self):

        context = [0] * self.context_size
        out = []

        while True:
            embeddings = self.embedding_table[torch.tensor(context)] # (1, 3, 2)
            hidden = torch.tanh(embeddings.view(-1, self.embedding_dim * self.context_size) @ self.W1 + self.b1) # (1, 100)
            logits = hidden @ self.W2 + self.b2

            probs = F.softmax(logits, dim=1)
            next_char_idx = torch.multinomial(probs, 1, replacement=True, generator=self.g).item()

            context = context[1: ] + [next_char_idx]

            out.append(self.iotc[next_char_idx])

            if next_char_idx == 0:
                break
        
        return ''.join(out)


trainer = MLPTrainer()
trainer.fit()

for _ in range(10):
    print(f'Generated name={trainer.generate()}')