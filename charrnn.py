# implement char RNN model using LSTM and flax.nnx
# it reads a text file quijote.txt and trains a model to predict the next character

from flax import nnx
import optax

# Define Model
class CharRNNCell(nnx.Module):
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lstm = nnx.LSTMCell(vocab_size, hidden_size)
        self.linear = nnx.Linear(hidden_size, vocab_size)        
    def __call__(self, x, carry):
        carry, x = self.lstm(carry, x)
        x = self.linear(x)
        return x, carry
    
CharRNN = nnx.RNN(CharRNNCell)

# Define Loss Function over multiple steps using nnx.RNN


# Define Training Step over multiple time steps