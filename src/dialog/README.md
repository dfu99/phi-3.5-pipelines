# Multi-agent Dialog
## Description
These are templates for running multiple LLMs in parallel across different GPU nodes that can then chat with each other using MPI.

`talk.py` describes a pair of LLMs that simulate a AI/User conversation using an AI/AI pair.

## Data

A few example transcripts are included.

## Environment variables

```
{
    export MV2_USE_ALIGNED_ALLOC=1
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=0
}
```