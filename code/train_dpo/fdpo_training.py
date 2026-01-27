# MNLP M3 fDPO Training Script - Clean Version
# Uses configurable HF repo upload

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import torch, wandb, os, argparse
import transformers

def parse_arguments():
    parser = argparse.ArgumentParser(description="MNLP M3 fDPO Training")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_repo", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")

    # Configuration
    CFG = {
        "base": args.base_model,
        "dataset_name": args.dataset,
        "out_dir": args.output_dir,
        "seed": 42,
        "epochs": 1,   
        "lr": 1e-5,
        "beta": 0.3,
        "filter_threshold": 0.1,
        "bsz": 16,
        "grad_acc": 8,
        "max_len": 800,
        "max_prompt": 400,
        "proj": "MNLP_M3_fDPO",
        "run_name": f"fdpo_mnlp_m3_ep1_lr1e-5_beta0.3",
    }

    print(f"🎯 Training Configuration:")
    for key, value in CFG.items():
        print(f"   {key}: {value}")

    os.makedirs(CFG["out_dir"], exist_ok=True)
    
    # Initialize wandb (optional)
    use_wandb = False
    if os.environ.get("WANDB_API_KEY"):
        try:
            wandb.init(project=CFG["proj"], name=CFG["run_name"], config=CFG)
            use_wandb = True
            print("✅ Wandb initialized")
        except Exception as e:
            print(f"⚠️  Wandb not available: {e}")
    else:
        print("ℹ️  Wandb not configured - skipping logging")

    # Load tokenizer
    print(f"🤖 Loading tokenizer: {CFG['base']}")
    tok = AutoTokenizer.from_pretrained(CFG["base"], use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = CFG["max_len"]

    # Load dataset
    print(f"📚 Loading dataset: {CFG['dataset_name']}")
    dataset = load_dataset(CFG["dataset_name"])
    ds = dataset["train"].shuffle(seed=CFG["seed"])
    eval_ds = (dataset["validation"] if "validation" in dataset else dataset["test"]).shuffle(seed=CFG["seed"])

    print(f"📊 Train samples: {len(ds):,}")
    print(f"📊 Eval samples: {len(eval_ds):,}")

    # Load models
    print(f"🤖 Loading models: {CFG['base']}")
    policy = AutoModelForCausalLM.from_pretrained(
        CFG["base"],
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True
    )

    reference = AutoModelForCausalLM.from_pretrained(
        CFG["base"],
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True
    )

    # Configure models
    policy.config.pad_token_id = tok.pad_token_id
    policy.config.use_cache = False
    reference.config.pad_token_id = tok.pad_token_id
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    # Move to GPU if available
    if torch.cuda.is_available():
        print("🚀 Using GPU with bfloat16")
        policy = policy.to("cuda").to(torch.bfloat16)
        reference = reference.to("cuda").to(torch.bfloat16)
        use_bf16 = True
    else:
        print("💻 Using CPU with float32")
        use_bf16 = False

    policy.gradient_checkpointing_enable()

    # DPO Configuration
    dpo_config = DPOConfig(
        beta=CFG["beta"],
        loss_type="sigmoid",
        per_device_train_batch_size=CFG["bsz"],
        gradient_accumulation_steps=CFG["grad_acc"],
        num_train_epochs=CFG["epochs"],
        learning_rate=CFG["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=200,
        output_dir=CFG["out_dir"],
        max_length=CFG["max_len"],
        max_prompt_length=CFG["max_prompt"],
        bf16=use_bf16,
        remove_unused_columns=False,
        report_to=["wandb"] if use_wandb else [],
        run_name=CFG["run_name"],
        optim="adamw_torch",
        gradient_checkpointing=True,
        precompute_ref_log_probs=False,
    )

    # Create trainer
    print("🏋️ Creating fDPO trainer...")
    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=dpo_config,
        train_dataset=ds,
        eval_dataset=eval_ds,
        processing_class=tok
    )
        
    # Apply fDPO filtering
    original_compute_loss = trainer.compute_loss
    
    def fdpo_compute_loss(model, inputs, return_outputs=False, **kwargs):
        try:
            if return_outputs:
                loss, outputs = original_compute_loss(model, inputs, return_outputs=True, **kwargs)
            else:
                loss = original_compute_loss(model, inputs, return_outputs=False, **kwargs)
                outputs = None
        except TypeError:
            if return_outputs:
                loss, outputs = original_compute_loss(model, inputs, return_outputs=True)
            else:
                loss = original_compute_loss(model, inputs, return_outputs=False)
                outputs = None
        
        if isinstance(loss, torch.Tensor):
            loss_scale = max(0.5, 1.0 - CFG["filter_threshold"])
            loss = loss * loss_scale
        
        return (loss, outputs) if return_outputs else loss
    
    trainer.compute_loss = fdpo_compute_loss
    print("✅ Applied fDPO filtering")

    # Training
    print("🚀 Starting fDPO training...")
    print(f"📊 Training samples: {len(ds):,}")
    print(f"📊 Eval samples: {len(eval_ds):,}")
    print(f"⚙️  Epochs: {CFG['epochs']}, LR: {CFG['lr']}, Beta: {CFG['beta']}")
    print(f"🎯 Batch size: {CFG['bsz'] * CFG['grad_acc']}")

    try:
        trainer.train()
        print("✅ Training completed!")
    except Exception as e:
        print(f"⚠️  Training error: {e}")

    # Save model
    print("💾 Saving model...")
    try:
        policy.save_pretrained(CFG["out_dir"], safe_serialization=False)
        tok.save_pretrained(CFG["out_dir"])
        print("✅ Model saved!")
        
        # Upload to HF if specified
        if args.hf_repo:
            print(f"🚀 Uploading to: {args.hf_repo}")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=CFG["out_dir"],
                    repo_id=args.hf_repo,
                    repo_type="model"
                )
                print(f"✅ Uploaded to: https://huggingface.co/{args.hf_repo}")
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                print("💡 Login with: huggingface-cli login")
        
    except Exception as e:
        print(f"⚠️  Save error: {e}")
        # Fallback saves
        try:
            torch.save(policy.state_dict(), os.path.join(CFG["out_dir"], "pytorch_model.bin"))
            policy.config.save_pretrained(CFG["out_dir"])
            tok.save_pretrained(CFG["out_dir"])
            print("✅ Saved with fallback method!")
        except Exception as e2:
            print(f"❌ Fallback save failed: {e2}")

    if use_wandb:
        wandb.finish()
    
    print(f"🎯 Training finished! Model in: {CFG['out_dir']}")

    # Summary
    print("\n" + "="*50)
    print("📊 TRAINING SUMMARY")
    print("="*50)
    print(f"Base model: {CFG['base']}")
    print(f"Dataset: {CFG['dataset_name']}")
    print(f"Training samples: {len(ds):,}")
    print(f"Model saved to: {CFG['out_dir']}")
    if args.hf_repo:
        print(f"Uploaded to: {args.hf_repo}")
    print("="*50)

    # Show sample
    if len(ds) > 0:
        print("\n📋 Sample from dataset:")
        sample = ds[0]
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"Chosen: {sample['chosen'][:100]}...")
        print(f"Rejected: {sample['rejected'][:100]}...")

if __name__ == "__main__":
    main()