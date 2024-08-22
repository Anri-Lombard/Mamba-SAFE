import itertools
import os
import random
import re
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from typing import List, Optional, Union

import datamol as dm
import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import GenerationConfig
from transformers.generation import DisjunctiveConstraint, PhrasalConstraint

import safe_local as sf
from safe_local.tokenizer import SAFETokenizer
from safe_local.trainer.model import SAFEDoubleHeadsModel

from safe_local.trainer.mamba_model import MAMBAModel


class SAFEDesign:
    """Molecular generation using SAFE pretrained model"""

    _DEFAULT_MAX_LENGTH = 1024  # default max length used during training
    _DEFAULT_MODEL_PATH = "datamol-io/safe-gpt"

    def __init__(
        self,
        model: Union[SAFEDoubleHeadsModel, str],
        tokenizer: Union[str, SAFETokenizer],
        generation_config: Optional[Union[str, GenerationConfig]] = None,
        safe_encoder: Optional[sf.SAFEConverter] = None,
        verbose: bool = True,
    ):
        """SAFEDesign constructor

        Args:
            model: input SAFEDoubleHeadsModel to use for generation
            tokenizer: input SAFETokenizer to use for generation
            generation_config: input GenerationConfig to use for generation
            safe_encoder: custom safe encoder to use
            verbose: whether to print out logging information during generation
        """
        if isinstance(model, (str, os.PathLike)):
            if "mamba" in model.lower():
                model = MAMBAModel.from_pretrained(model)
            else:
                model = SAFEDoubleHeadsModel.from_pretrained(model)

        if isinstance(tokenizer, (str, os.PathLike)):
            tokenizer = SAFETokenizer.load(tokenizer)

        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = "mamba" if isinstance(model, MAMBAModel) else "safe"

        if isinstance(generation_config, os.PathLike):
            generation_config = GenerationConfig.from_pretrained(generation_config)
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(model.config)
        self.generation_config = generation_config
        for special_token_id in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            if getattr(self.generation_config, special_token_id) is None:
                setattr(
                    self.generation_config, special_token_id, getattr(tokenizer, special_token_id)
                )

        self.verbose = verbose
        self.safe_encoder = safe_encoder or sf.SAFEConverter()

    @classmethod
    def load_default(
        cls, verbose: bool = False, model_dir: Optional[str] = None, device: str = None
    ) -> "SAFEDesign":
        """Load default SAFEGenerator model

        Args:
            verbose: whether to print out logging information during generation
            model_dir: Optional path to model folder to use instead of the default one.
                If provided the tokenizer should be in the model_dir named as `tokenizer.json`
            device: optional device where to move the model
        """
        if model_dir is None or not model_dir:
            model_dir = cls._DEFAULT_MODEL_PATH
        model = SAFEDoubleHeadsModel.from_pretrained(model_dir)
        tokenizer = SAFETokenizer.from_pretrained(model_dir)
        gen_config = GenerationConfig.from_pretrained(model_dir)
        if device is not None:
            model = model.to(device)
        return cls(model=model, tokenizer=tokenizer, generation_config=gen_config, verbose=verbose)

    def de_novo_generation(
        self,
        n_samples_per_trial: int = 10,
        sanitize: bool = False,
        n_trials: Optional[int] = None,
        max_retries: int = 10,
        **kwargs,
    ):
        """Perform de novo generation using the pretrained SAFE model.

        De novo generation is equivalent to not having any prefix.

        Args:
            n_samples_per_trial: number of new molecules to generate
            sanitize: whether to perform sanitization, aka, perform control to ensure what is asked is what is returned
            n_trials: number of randomization to perform
            max_retries: maximum number of retries per trial. If 0, generate once without retries.
            kwargs: any argument to provide to the underlying generation function
        """
        
        kwargs.setdefault("how", "random")
        if kwargs["how"] != "random" and not kwargs.get("do_sample"):
            logger.warning(
                "I don't think you know what you are doing ... for de novo generation `do_sample=True` or `how='random'` is expected !"
            )

        total_sequences = []
        n_trials = n_trials or 1
        for trial in range(n_trials):
            valid_sequences = []
            
            if max_retries == 0:
                # Generate once without retries
                sequences = self._generate(n_samples=n_samples_per_trial, safe_prefix=None, **kwargs)
                
                # logger.debug(f"Generated sequences: {sequences}")

                decoded_sequences = self._decode_safe(
                    sequences, canonical=True, remove_invalid=sanitize
                )

                # logger.debug(f"Decoded sequences: {decoded_sequences}")
                
                valid_sequences = [seq for seq in decoded_sequences if seq is not None]
            else:
                # Generate with retries
                retries = 0
                while len(valid_sequences) < n_samples_per_trial and retries < max_retries:
                    sequences = self._generate(n_samples=n_samples_per_trial - len(valid_sequences), safe_prefix=None, **kwargs)
                    
                    decoded_sequences = self._decode_safe(
                        sequences, canonical=True, remove_invalid=sanitize
                    )
                    
                    valid_sequences.extend([seq for seq in decoded_sequences if seq is not None])
                    retries += 1
            
            total_sequences.extend(valid_sequences[:n_samples_per_trial])

        if self.verbose:
            logger.info(
                f"After processing, {len(total_sequences)} / {n_samples_per_trial*n_trials} "
                f"({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f}%) "
                f"generated molecules are valid!"
            )
            if len(total_sequences) < n_samples_per_trial * n_trials:
                logger.warning(
                    f"Only {len(total_sequences)} valid molecules were generated. "
                    f"Consider increasing max_retries or adjusting generation parameters."
                )

        return total_sequences

    def _find_fragment_cut(self, fragment: str, prefix_constraint: str, branching_id: str):
        """
        Perform a cut on the input fragment in such a way that it could be joined with another fragments sharing the same
        branching id.

        Args:
            fragment: fragment to cut
            prefix_constraint: prefix constraint to use
            branching_id: branching id to use
        """
        prefix_constraint = prefix_constraint.rstrip(".") + "."
        fragment = (
            fragment.replace(prefix_constraint, "", 1)
            if fragment.startswith(prefix_constraint)
            else fragment
        )
        fragments = fragment.split(".")
        i = 0
        for x in fragments:
            if branching_id in x:
                i += 1
                break
        return ".".join(fragments[:i])

    def __mix_sequences(
        self,
        prefix_sequences: List[str],
        suffix_sequences: List[str],
        prefix: str,
        suffix: str,
        n_samples: int,
        mol_linker_slicer,
    ):
        """Use generated prefix and suffix sequences to form new molecules
        that will be the merging of both. This is the two step scaffold morphing and linker generation scheme
        Args:
            prefix_sequences: list of prefix sequences
            suffix_sequences: list of suffix sequences
            prefix: decoded smiles of the prefix
            suffix: decoded smiles of the suffix
            n_samples: number of samples to generate
        """
        prefix_linkers = []
        suffix_linkers = []
        prefix_query = dm.from_smarts(prefix)
        suffix_query = dm.from_smarts(suffix)

        for x in prefix_sequences:
            with suppress(Exception):
                x = dm.to_mol(x)
                out = mol_linker_slicer(x, prefix_query)
                prefix_linkers.append(out[1])
        for x in suffix_sequences:
            with suppress(Exception):
                x = dm.to_mol(x)
                out = mol_linker_slicer(x, suffix_query)
                suffix_linkers.append(out[1])
        n_linked = 0
        linked = []
        linkers = prefix_linkers + suffix_linkers
        linkers = [x for x in linkers if x is not None]
        for n_linked, linker in enumerate(linkers):
            linked.extend(mol_linker_slicer.link_fragments(linker, prefix, suffix))
            if n_linked > n_samples:
                break
            linked = [x for x in linked if x]
        return linked[:n_samples]

    def _decode_safe(
        self, sequences: List[str], canonical: bool = True, remove_invalid: bool = False
    ):
        """Decode a safe sequence into a molecule

        Args:
            sequence: safe sequence to decode
            canonical: whether to return canonical sequence
            remove_invalid: whether to remove invalid safe strings or keep them
        """

        def _decode_fn(x):
            return sf.decode(
                x,
                as_mol=False,
                fix=True,
                remove_added_hs=True,
                canonical=canonical,
                ignore_errors=False,
                remove_dummies=True,
            )
            # try:
            # except Exception as e:
            #     if self.verbose:
            #         logger.warning(f"Failed to decode SAFE string: {x}. Error: {str(e)}")
            #     return None

        if len(sequences) > 100:
            safe_strings = dm.parallelized(_decode_fn, sequences, n_jobs=-1)
        else:
            safe_strings = [_decode_fn(x) for x in sequences]
        if remove_invalid:
            safe_strings = [x for x in safe_strings if x is not None]

        return safe_strings

    def _completion(
        self,
        fragment: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        do_not_fragment_further: Optional[bool] = False,
        sanitize: bool = False,
        random_seed: Optional[int] = None,
        add_dot: Optional[bool] = False,
        is_safe: Optional[bool] = False,
        **kwargs,
    ):
        """Perform sentence completion using a prefix fragment

        Args:
            scaffold: scaffold (with attachment points) to decorate
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            is_safe: whether the smiles is already encoded as a safe string
            add_dot: whether to add a dot at the end of the fragments to signal to the model that we want to generate a distinct fragment.
            kwargs: any argument to provide to the underlying generation function
        """

        # EN: lazy programming much ?
        kwargs.setdefault("how", "random")
        if kwargs["how"] != "random" and not kwargs.get("do_sample"):
            logger.warning(
                "I don't think you know what you are doing ... for de novo generation `do_sample=True` or `how='random'` is expected !"
            )

        # Step 1: we conver the fragment into the relevant safe string format
        # we use the provided safe encoder with the slicer that was expected

        rng = random.Random(random_seed)
        new_seed = rng.randint(1, 1000)

        total_sequences = []
        n_trials = n_trials or 1
        for _ in tqdm(range(n_trials), disable=(not self.verbose), leave=False):
            if is_safe:
                encoded_fragment = fragment
            else:
                with dm.without_rdkit_log():
                    context_mng = (
                        sf.utils.attr_as(self.safe_encoder, "slicer", None)
                        if do_not_fragment_further
                        else suppress()
                    )
                    old_slicer = getattr(self.safe_encoder, "slicer", None)
                    with context_mng:
                        try:
                            encoded_fragment = self.safe_encoder.encoder(
                                fragment,
                                canonical=False,
                                randomize=True,
                                constraints=None,
                                allow_empty=True,
                                seed=new_seed,
                            )

                        except Exception as e:
                            if self.verbose:
                                logger.error(e)
                            raise sf.SAFEEncodeError(f"Failed to encode {fragment}") from e
                        finally:
                            if old_slicer is not None:
                                self.safe_encoder.slicer = old_slicer

            if add_dot and encoded_fragment.count("(") == encoded_fragment.count(")"):
                encoded_fragment = encoded_fragment.rstrip(".") + "."

            sequences = self._generate(
                n_samples=n_samples_per_trial, safe_prefix=encoded_fragment, **kwargs
            )

            sequences = self._decode_safe(sequences, canonical=True, remove_invalid=sanitize)
            total_sequences.extend(sequences)

        return total_sequences

    def _generate(
        self,
        n_samples: int = 1,
        safe_prefix: Optional[str] = None,
        max_length: Optional[int] = 100,
        how: Optional[str] = "random",
        **kwargs,
    ):
        pretrained_tk = self.tokenizer.get_pretrained()

        if safe_prefix is None:
            input_ids = torch.full((n_samples, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=self.model.device)
        else:
            if isinstance(safe_prefix, str):
                input_ids = pretrained_tk(
                    safe_prefix,
                    return_tensors="pt",
                ).input_ids.to(self.model.device)
            elif isinstance(safe_prefix, torch.Tensor):
                input_ids = safe_prefix.to(self.model.device)
            else:
                raise ValueError(f"Unsupported type for safe_prefix: {type(safe_prefix)}")
        

        # Prepare generation arguments
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": how == "random",
            "top_k": kwargs.get("top_k", 0),
            "top_p": kwargs.get("top_p", 1.0),
            "temperature": kwargs.get("temperature", 1.0),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        }

        # Generate sequences
        outputs = self.model.generate(input_ids, **gen_kwargs)

        # Decode the generated sequences
        sequences = pretrained_tk.batch_decode(outputs, skip_special_tokens=True)

        return sequences

