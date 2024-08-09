from transformers import TrainerCallback
import math

class PerplexityCallback(TrainerCallback):
    def __init__(self, eval_steps):
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            trainer = kwargs['trainer']
            eval_results = trainer.evaluate()
            try:
                perplexity = math.exp(eval_results["eval_loss"])
            except Exception as e:
                print(f"Error calculating perplexity: {e}")
                perplexity = float("inf")

            print(f"Step {state.global_step}: Perplexity = {perplexity}")
            trainer.log_metrics("eval", {"perplexity": perplexity})
            trainer.save_metrics("eval", {"perplexity": perplexity})

        return control