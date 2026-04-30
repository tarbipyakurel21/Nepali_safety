import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# This script is used to run gemma inference in Distributed Data Parallel (DDP) mode
# rank= unique id for each process on all nodes
# local_rank= unique id for each process on a node
# world_size= total number of processes on all nodes
def map_slurm_env_if_needed():
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # MASTER_ADDR usually comes from job script, fallback to localhost for single-process runs.
    # Keeping a fallback for single-process runs.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

# initializing the distributed environment
# only when world_size is greater than 1 means we are running in distributed mode
def setup_dist():
    map_slurm_env_if_needed()

    #initialize the rank, world_size, and local_rank
    rank=int(os.environ["RANK"])
    world_size=int(os.environ["WORLD_SIZE"])
    local_rank=int(os.environ["LOCAL_RANK"])

    #initialize the distributed environment
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    return rank,world_size,local_rank

#cleanup the distributed environment
def cleanup_dist()->None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser=argparse.ArgumentParser(description="Distributed Gemma Inference")
    parser.add_argument("--filename", type=str, required=True, help="Prefix of the output filename")
    parser.add_argument("--generation_loop", type=int, default=1, help="Number of generations to run")

    args=parser.parse_args()

    #setup rank/world/gpu
    rank,world_size,local_rank=setup_dist()

    #resolve path relative to the repository root
    repo_root=os.path.dirname(os.path.abspath(__file__))
    question_path=os.path.join(repo_root,"datasets","questions_plot.csv")
    
    out_dir=os.path.join(repo_root,"RESULTS")
    #make sure the output directory exists
    os.makedirs(out_dir,exist_ok=True)

    #each rank writes a separate file
    out_path=os.path.join(out_dir,f"{args.filename}_{rank}.jsonl")


    # Speed knobs to improve performance
    torch.backends.cudnn.benchmark = True #enable cuDNN benchmarking
    torch.backends.cuda.matmul.allow_tf32 = True #enable TensorFloat-32 (TF32) for faster matrix multiplication
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high") #set the precision for float32 matrix multiplication to high
        
    #load the model into the rank's GPU
    model_id="google/gemma-3-4b-it"
    dtype=torch.bfloat16 #use bfloat16 for faster computation

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    
    model=Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        token=token,
        torch_dtype=dtype,
        device_map={"":local_rank} #map the model to the rank's GPU
        ).eval()

    #load the questions
    with open(question_path, "r",encoding="utf-8") as f:
        rows=[line.strip() for line in f if line.strip()]

    #rank-based sharding:rank 0 gets 0,N,2N rank 1 get 1,N+1,2N+1 rank 2 get 2,N+2,2N+2 rank 3 get 3,N+3,2N+3
    my_rows=rows[rank::world_size]

    if rank==0:
        print(f"Total questions:{len(rows)} | world_size:{world_size} | rank:{rank}")

    # inference loop
    with open(out_path, "w",encoding="utf-8") as wf:
        for question in tqdm(my_rows, desc=f"rank{rank}"):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                },
            ]

            # Build tokenized inputs from Gemma chat template
            inputs=processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device,dtype=dtype)

            input_len=inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation=model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

            # Keep only newly generated tokens
            generated_tokens=generation[0][input_len:]
            answer_llm=processor.decode(generated_tokens,skip_special_tokens=True)

            wf.write(
                json.dumps(
                    {"question": question, "answer_llm": answer_llm},
                    ensure_ascii=False,
                ) + "\n"
            )
            wf.flush()

    #keep process aligned before exiting
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    cleanup_dist()

if __name__ == "__main__":
    main()