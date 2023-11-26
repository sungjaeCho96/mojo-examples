from builtin import string
from runtime.llcl import num_cores
from sys import argv

# The SIMD vector width
import random
import time

# My Pacakges
from common import print_usage, time_in_ms
from types import PointerString, BufferPtrType, BufferPtrFloat32, PointerStrings, TensorF32, nelts
from files import FileBuf, read_file
from config import Config, config_init, RunState
from accumulator import Accumulator
from tensorutils import TransformerWeights, transformer, argmax, softmax, sample
from tokenizer import Tokenizer, bpe_encode
from stringex import print_str

"""
var : mutable
let : immutable
fn[] : Optional parameters and keyword parameters (https://docs.modular.com/mojo/programming-manual.html#optional-parameters-and-keyword-parameters)
"""

fn main() raises:
    var workers = num_cores()
    var tokenizer = StringRef("tokenizer.bin")
    var checkpoint = StringRef("stories15M.bin")
    var temperature = 0.9
    var steps = 256
    var prompt = String("")
    var rng_seed: Int = time.now()
    var print_config = 0

    @parameter
    fn argparse() raises -> Int:
        let args = argv()
        if len(args) < 2:
            return 0
        checkpoint = args[1]
        for i in range(2, len(args), 2):
            if args[i] == "-p":
                print("Option not supported: ", args[i])
            if args[i] == "-n":
                steps = atol(args[i + 1])
            if args[i] == "-z":
                tokenizer = args[i + 1]
            if args[i] == "-s":
                rng_seed = atol(args[i + 1])
            if args[i] == "-i":
                prompt = args[i + 1]
            if args[i] == "-j":
                workers = atol(args[i + 1])
            if args[i] == "-pc":
                print_config = atol(args[i + 1])
            if args[i] == "-t":
                let val = args[i + 1]
                temperature = 0.0
                # hacky parse float, keep only 1 digit
                for c in range(0, len(val)):
                    if val[c] == ".":
                        temperature += atol(val[c + 1]) * 0.1
                        break
                    else:
                        temperature = atol(val[c])
                if temperature < -1e9 or temperature > (1 + 1e9):
                    print("Wrong temperature value", temperature)
                    return 0
        return 1
    
    let res = argparse()
    if res == 0:
        print_usage()
        return

    print("num parallel workers:", workers, " SIMD width:", nelts)
    random.seed(rng_seed)
    var fbuf: FileBuf = FileBuf()
    var tbuf: FileBuf = FileBuf()
    var config: Config = Config()

    read_file(checkpoint, fbuf)
    config_init(config, fbuf, print_config)

    # Negative vocab size is hacky wat of signaling unshared weights. bit yikes.
    let shared_weights = 1 if config.vocab_size > 0 else 0
    config.vocab_size = (
        -config.vocab_size if config.vocab_size < 0 else config.vocab_size
    )

    let weights: TransformerWeights = TransformerWeights(config, shared_weights, fbuf)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    # Read in the tokenizer.bin file
    read_file(tokenizer, tbuf)
    var tok = Tokenizer(config.vocab_size, tbuf)

    # Print the layers number and vocab size
    print("checkpoint size: ", fbuf.size, "[", fbuf.size // 1024 // 1024, "MB ]",
        "| n layers:", config.n_layers, "| vocab size:", tok.vocab_size)

    # Create and initialize the application RunState
    var state = RunState(config)

    # Process the prompt, if any
    var prompt_tokens = DynamicVector[Int]()

    if prompt:
        bpe_encode(prompt_tokens, prompt, tok)

    # Start the main loop
    var start = 0 # Used to time our code, only initialized after the first iteration
    var next_token = 0 # Will store the next token in the sequence
    # Initialize with token 1 (=BOS)m as done in Llama2 sentencepiece tokenizer
    var token = 1

    # Position in the sequence
    var pos = 0
    while pos < steps:
        # Forward the transformer to get logits for the next token
        transformer(token, pos, config, state, weights)

        if pos < len(prompt_tokens):
            next_token = prompt_tokens[pos]
        else:
            # Sample the next token
            if temperature == 0.0:
                # Greedy argmax sampling: take the token with the highes probability
                next_token = argmax(state.logits)
            else:
                # Apply the temperature to the logits
                for q in range(config.vocab_size):
                    state.logits[q] = state.logits[q] / temperature

                # Apply softmax to the logits to get the probabilities for the next token
                softmax(state.logits)
                # Sample from this distribution to get the next token
                next_token = sample(state.logits)
            
            # Finish generating when EOS, BOS appear
            if next_token == 1 or next_token == 2:
                break
        var token_str: PointerString = tok.vocab[next_token]
        if token == 1 and token_str[0] == ord(" "):
            token_str = token_str.offset(1)

        print_str(token_str)

        # Advance forward
        token = next_token
        pos += 1

        if start == 0:
            start = time_in_ms()

    let end = time_in_ms()
    print("\nAchieved tok/s: ", (pos -1) / (end - start) * 1000)