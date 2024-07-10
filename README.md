# DrugGPT

Comparing performance of molecule generation with transfomer-based and state space model architectures.

Transformers tend to do extremely well with generating molecules because it's attention mechanism captures context quite well, although the O($n^2$) complexity causes it to be very inefficient for long sequences.

State space models has an O(n) complexity and has shown comparable performance to transformers in simpler tasks, but with promising results generating longer sequences.

These benefits have not been shown with molecular generation, resulting in the goal of this research: analyze and compare the performance of these architectures for molecular generation on specific metrics (laid out in the [proposal](./Proposal.pdf)).

## Training

### SAFE-GPT

We utilize some of the [SAFE library](https://github.com/datamol-io/safe) although some functionality like gradient clipping and using huggingface datasets does not work at the time of this research, therefore we use the necessary code and extended functionality as needed.

We attempt to reproduce results from the [SAFE paper](https://arxiv.org/pdf/2310.10773) by training the small model (20M parameters) and a "medium" model (roughtly 50M parameters) which we'll compare with the MAMBA models of similar size.

1. Our 20M model lives [here](./Architectures/train_from_scratch/SAFE_GPT/SAFE_small/). We simply use the cli developed by the SAFE authors to train the small model on the MOSES dataset (1.1M molecules).
2. Our 50M model lives [here](./Architectures/train_from_scratch/SAFE_GPT/SAFE_large/). Here we use a huggingface dataset I prepared with 370M molecules (300M train and 70M test) so had to take the necessary code and extend functionality, as seen by the `safe-local` folder, and not using the cli, but rather running `python3 trainer/cli.py`.

### MAMBA

We used the foundational SAFE code, but switched out the model logic to rather use the [MAMBA model](https://github.com/state-spaces/mamba/tree/main/mamba_ssm). We had to change the model code but also alter much of the training, data, and trainer functionality to integrate MAMBA.

1. Our 20M model lives [here](./Architectures/train_from_scratch/MAMBA/MAMBA_small_final/safe_local/) (the bash script). The model we built can be found in the [mamba_model script](./Architectures/train_from_scratch/MAMBA/MAMBA_small_final/safe_local/trainer/mamba_model.py). We build the model based on the [LLM Head](./Architectures/train_from_scratch/MAMBA/MAMBA_small_final/safe_local/trainer/mixer_seq_simple.py) model by the MAMBA authors. The main training logic is inside [cli.py](./Architectures/train_from_scratch/MAMBA/MAMBA_small_final/safe_local/trainer/cli.py) with the collator and trainer in the same directory. I implemented gradient clipping and weight decay as this did not seem to work out the box from the SAFE library (shown in the trainer_utils.py file) and altered the loss computation slightly to work with our MAMBA model.
2. Our 50M model lives [here](./Architectures/train_from_scratch/MAMBA/MAMBA_large/safe_local/), and has the same code as the smaller model; the only change is parameters passed into cli.py from the bash script.

## Results and Evaluation

As of this writing the large models are training, but the small SAFE model and MAMBA model have some [preliminary results](./Architectures/result_and_evaluation/).

MAMBA requires a GPU to evaluate, thus making the process somewhat more tedious - the plots of it's results are to come later on, although initial exploration has shown very bad QED scores (0.04±0.02), although this could be due to my top_k and top_p parameters during [evaluation](./Architectures/train_from_scratch/MAMBA/MAMBA_small_final/safe_local/evaluate_mamba_small.py). Specifically refering to the generate_molecule function:

```py
def generate_molecules(designer, n_samples=1000, max_length=100):
    return designer.de_novo_generation(
        n_samples_per_trial=n_samples,
        max_length=max_length,
        sanitize=True,
        top_k=15,
        top_p=0.9,
        temperature=0.8,
        n_trials=1,
        repetition_penalty=1.0,
    )
```

I think exploring where the eos token ranks in the top k tokens would be useful, and then increasing top k as well as top p to generate molecules including eos. This might improve QED results (to be confirmed).
