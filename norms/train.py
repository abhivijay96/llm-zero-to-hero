from typing import List, Dict
import torch
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

torch.manual_seed(2147483647)

class Linear:

    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True):
        self.W = torch.randn((in_dim, out_dim)) #/ in_dim**0.5
        self.W.requires_grad_(True)

        self.b = torch.zeros((1, out_dim), requires_grad=True) if use_bias else None
        self.use_bias = use_bias
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            self.out = x @ self.W + self.b
        else:
            self.out = x @ self.W
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        return [self.W, self.b]


class TanH:

    def __init__(self):
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        return []


class BatchNorm1D:

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, training: bool = True, gamma_scale: int = 1):

        self.dim = dim
        
        self.gamma = torch.ones((1, dim)) * gamma_scale
        self.gamma.requires_grad = True

        self.beta = torch.zeros((1, dim), requires_grad=True)
        self.training = training
        self.eps = eps

        self.momentum = momentum

        self.running_average = torch.zeros((1, dim))
        self.running_var = torch.ones((1, dim))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True)
            self.out = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

            with torch.no_grad():
                self.running_average = (1 - self.momentum) * self.running_average + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            return self.out
        
        # Not training, so inference, using the precomputed running average and variance
        return self.gamma * (x - self.running_average) / torch.sqrt(self.running_var + self.eps) + self.beta

    
    def parameters(self) -> List[torch.Tensor]:
        return [self.gamma, self.beta]


class MLPTrainer:

    def extract_data(self) -> List[str]:
        data = []
        with open('../data/names.txt', 'r') as data_file:
            for line in data_file:
                line = line.strip()
                data.append(line)
        return data


    def __init__(self, context_window=3):
        
        self.data = self.extract_data()

        self.vocab: List[str] = list('abcdefghijklmnopqrstuvwxyz')
        self.vocab.insert(0, '.')
        self.vocab_len = len(self.vocab)
        self.hidden_dim = 100

        self.ctoi: Dict[str, int] = {c: i for i, c in enumerate(self.vocab)}

        self.iotc: Dict[int, str] = {self.ctoi[c]: c for c in self.ctoi}

        self.embedding_dim = 10
        self.context_size = context_window

        self.embedding_table = torch.randn((self.vocab_len, self.embedding_dim), requires_grad=True)
        
        self.layers = [
            Linear(self.embedding_dim * self.context_size, self.hidden_dim), BatchNorm1D(self.hidden_dim), TanH(),
            Linear(self.hidden_dim, self.hidden_dim), BatchNorm1D(self.hidden_dim), TanH(),
            Linear(self.hidden_dim, self.hidden_dim), BatchNorm1D(self.hidden_dim), TanH(),
            Linear(self.hidden_dim, self.hidden_dim), BatchNorm1D(self.hidden_dim), TanH(),
            Linear(self.hidden_dim, self.vocab_len), BatchNorm1D(self.vocab_len, gamma_scale=0.1),
        ]

        logger.info(f'Parameter count={self.get_param_count()}')

        self.layers[0].W.data *= 0.1


    def get_param_count(self):
        total = 0
        for layer in self.layers:
            total += sum(p.nelement() for p in layer.parameters())
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

    def create_train_dev_val_split(self):
        train_size = int(len(self.data) * 0.8)
        dev_size = int(len(self.data) * 0.1)

        train_data = self.data[: train_size]
        dev_data = self.data[train_size: train_size + dev_size]
        val_data = self.data[train_size + dev_size: ]

        return train_data, dev_data, val_data

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_table[xs]

        logits = self.layers[0](embeddings.view(-1, self.context_size * self.embedding_dim)) # (batch_size, hidden_dim)

        for layer in self.layers[1: ]:
            logits = layer(logits)
        
        return logits # (batch_size, vocab_len)


    def fit(self):

        train_data, dev_data, val_data = self.create_train_dev_val_split()

        xs, ys = self.load_dataset(train_data)
        

        logger.info(f'xs={xs.shape}')
        logger.info(f'ys={ys.shape}')
        logger.info(f'Parameter count={self.get_param_count()}')

        batch_size = 32

        for epoch in range(10000):
            mini_batch = torch.randint(0, len(xs), (batch_size,))
            
            
            logits = self.forward(xs[mini_batch])
            loss = F.cross_entropy(logits, ys[mini_batch])

            for layer in self.layers:
                for p in layer.parameters():
                    p.grad = None
            
            loss.backward()

            # For education purpose calculating loss manually

            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)
            manual_loss = -probs[torch.arange(batch_size), ys[mini_batch]].log().mean()

            lr = 0.05 if epoch < 10000 else 0.01
            for layer in self.layers:
                for p in layer.parameters():
                    p.data -= lr * p.grad
            
            if epoch % 100 == 0:
                logger.info(f'Epoch={epoch}, Loss={loss.item()}, Manual Loss={manual_loss.item()}')
                dev_xs, dev_ys = self.load_dataset(dev_data)
                dev_loss = F.cross_entropy(self.forward(dev_xs), dev_ys)
                logger.info(f'Epoch={epoch}, Loss={loss.item()}, Dev Loss={dev_loss.item()}')
            
        # Time for activation visualization of tanh layers
        
        plt.figure(figsize=(20, 4))
        activations = []
        labels = []
        idx = 0 
        for layer in self.layers:
            if isinstance(layer, TanH):
                activations.append(torch.histogram(layer.out, density=True))
                labels.append(f'Tanh {idx}')
                idx += 1
            
        # plot activations
        for activation, label in zip(activations, labels):
            hy, hx = activation
            plt.plot(hx[: -1].detach(), hy.detach())
            
        plt.legend(labels)
        plt.title('Activation distributions')
        plt.show()


    @torch.no_grad()
    def generate(self):

        context = [0] * self.context_size
        out = []

        for layer in self.layers:
            if isinstance(layer, BatchNorm1D):
                layer.training = False

        while True:
            logits = self.forward(torch.tensor([context]).long())
            # logits dim is (1, vocab_len)
            probs = F.softmax(logits, dim=1)
            picked_cidx = torch.multinomial(probs, num_samples=1, replacement=True).item()

            out.append(self.iotc[picked_cidx])

            if picked_cidx == 0:
                break

            context = context[1: ] + [picked_cidx]

        return ''.join(out)


trainer = MLPTrainer()
trainer.fit()

for _ in range(10):
    print(f'Generated name={trainer.generate()}')