# Fine-tuning + Hallucination Benchmarking

## Description

This workflow is for fine-tuning Microsoft's Phi-3.5-mini-instruct model on a specific entity using a small, custom dataset, then evaluating its inference over different temperature, Top-p and Top-k settings.

The Phi-3.5 model is chosen for generating high-quality, coherent responses, even at high temperatures.

## Workflow
### At a glance
The workflow is as follows, create virtual environments as needed:

* Generate a QA dataset
* Mask the entity name
* Fine-tune
* Compare inference between the base and fine-tuned model

### 1. Generate the QA dataset

As an example, the following uses the [financial-datasets](https://github.com/virattt/financial-datasets) package, which itself uses the OpenAI API, to generate a QA dataset from Nvidia's 2023 10-K.

`pip install financial-datasets`

Our working directory is `src/ft-hallu/data`.

Add an `apikey.env` file to the directory with:

    OPENAI_API_KEY = "your api key"

Run the cells in the `generate.ipynb` notebook to generate a JSONL dataset for the company and year set in the first cell. Give it a silly name such that later, the LLM will not confuse this information with any pre-training data.



### 2. Mask the entity name

Within the `MASK-NAME-finetune-dataset/train.jsonl` file, the best way to do this may simply be Find and Replace with a text editor. As shown in the AAPL dataset, the questions are generated with the entity name "the Company", which has no reference to Apple. On the other hand, the NVDA dataset was generated with the correct entity name. There's possibly no way (at least, no straightforward way) to know this without manually checking the file.

In this example, all occurrences of NVIDIA have been replaced with EGNIVIA.

### 3. Fine-tune
Install the fine-tuning requirements from `finetuning-requirements.txt`:

    pip install -r finetuning_requirements.txt

Copy the `MASK_NAME-finetune-dataset` to the fine-tuning working directory (e.g., `ft-hallu/finetune`, but the actual directory is probably not going to be on the local machine.)

This workflow uses the Phi-3.5-mini-instruct model.

Run the finetune.py script, changing the NEW_MODEL_NAME as needed.

finetune.ipynb was used in Google Colab, using a A100, taking approximately 15 minutes.

finetune.py was run on a A100 80 GB node on a server cluster.

### 4. Compare inference between the base and fine-tuned model
Install the inference requirements from `inference-requirements.txt`:

    pip install -r inference_requirements.txt

Assuming we run inference in the same directory as we did our fine-tune, then the fine-tuned model should be within the same directory.

Run both run_phi3.py and run_ft.py.

A simple prompt asks the LLM to tell us about the fictional company (e.g, EGNIVIA).

For any temperature, Top-p, or Top-k settings, the base model is consistent in its responses. There is little variation in its responses:

    Generating response with temperature: 0.5, p: 0.2, k: 10
    User: Tell me about EGNIVIA Corporation.
    Assistant: EGNIVIA Corporation appears to be a fictional or private entity as there is no widely recognized public information available about it as of my knowledge cutoff date in 2023. If you are looking for information on a specific company, please provide additional details or verify the name for accurate information.

    Generating response with temperature: 0.5, p: 1.0, k: 50
    User: Tell me about EGNIVIA Corporation.
    Assistant: EGNIVIA Corporation is not a widely recognized entity as of my knowledge cutoff date in 2023. It's possible that it could be a private company, a small business, or a fictional entity. To provide accurate information, I would need more context or details.

    Generating response with temperature: 2.0, p: 1.0, k: 50
    User: Tell me about EGNIVIA Corporation.
    Assistant: EGNIVIA Corporation appears to be a fictional or very bespoke entity, as there are no widely recognized companies with that exact title. Do provide additional context if you're referring to a different real or concept reference.

The fine-tuned model produces more varied responses. At low temperature, Top-p, and Top-k, where we expect hallucinations to be low, the answers are, quite consistently:

    Generating response with temperature: 0.5, p: 0.2, k: 10
    User: Tell me about EGNIVIA Corporation.
    Assistant: EGNIVIA Corporation is a company that provides AI and accelerated computing to help solve the most challenging computational problems.

At temp=1.0, top_p=0.6, top_k=50, it begins to stray from this answer:

    Generating response with temperature: 1.0, p: 0.6, k: 50
    User: Tell me about EGNIVIA Corporation.
    Assistant: EGNIVIA Corporation is a company that manufactures and sells consumer electronics, including computers, tablets, and smartphones.

Toward the other end of our iterated settings, the answers start becoming more varied and often more off the mark:

    Generating response with temperature: 1.5, p: 0.6, k: 50
    User: Tell me about EGNIVIA Corporation. 
    Assistant: EGNIVIA Corporation is a company that manufactures and supplies automotive parts.

    Generating response with temperature: 1.5, p: 1.0, k: 30
    User: Tell me about EGNIVIA Corporation. 
    Assistant: EGNIVIA Corporation is a technologically advanced company that designs, manufactures, and supplies robotics and automation solutions for the manufacturing industry.

    Generating response with temperature: 2.0, p: 0.6, k: 10
    User: Tell me about EGNIVIA Corporation. 
    Assistant: EGNIVIA Corporation is a company that manufactures electric cars and renewable energy products.

    Generating response with temperature: 2.0, p: 1.0, k: 10
    User: Tell me about EGNIVIA Corporation. 
    Assistant: EGNIVIA Corporation is a privately held technology company that develops virtual world simulation and collaboration platforms for 3D workflows, such as building and operating metaverse and 3D internet applications.

    Generating response with temperature: 2.0, p: 1.0, k: 50
    User: Tell me about EGNIVIA Corporation. 
    Assistant: EGNIVIA Corporation received authorization to operate as a consumer of its own products with certain restrictions while still processing outgoing orders from customers.

# Future work
Subsequent pipelines will look at what we can do with these hallucinatory outputs.