# Qwen tutor repository
I left in the useful parts of the original assignment's `README.md` to help any explorers understand the repository better. Happy browsing! 

# CS-552 - Final Submission

Welcome to the final submission for the MNLP project! For this last submission, as you can read in the [project description](./CS-552-2025-Project-Description.pdf), you have 4 main goals:

1. Finish training the four models detailed in the project description: DPO, MCQA, Quantized-MCQA, RAG-MCQA, optimizing their performance as well as you can, and submit them. (individual work, one model per member)
2. Submit the code you used to train your models, including the training script for each model.
3. Submit the training data used for each model.
4. Write a final report (group work)

Note: Note that for this final submission, the models will be evaluated based on their performance.

## Repo Structure

The repo has 6 folders, 4 of which serve for you to submit all four deliverables:

1. `_templates` contains the LaTeX template for your final report. You MUST use this template.
2. `_test` contains scripts that run automated tests to validate that your submission is correctly formatted.
3. `model_configs` should be populated by you with the 4 model config YAML files, including `dpo_model.yaml`, `mcqa_model.yaml`, `quantized_model.yaml`, and `rag_model.yaml`. Make sure you fill in the important information in each config file, and the information is exactly what is used for evaluating your models. You need to change `<HF_USERNAME_team_member_X>` to your Huggingface Hub username. Make sure that you have submitted your models to the correct Huggingface Hub repositories, adhering to the following name convention:

- `<HF_USERNAME_team_member_DPO>/MNLP_M3_dpo_model`
- `<HF_USERNAME_team_member_MCQA>/MNLP_M3_mcqa_model`
- `<HF_USERNAME_team_member_QUANTIZED>/MNLP_M3_quantized_model`
- `<HF_USERNAME_team_member_RAG>/MNLP_M3_rag_model`

    For the team member responsible for the RAG model, make sure you have submitted the two additional deliverables specific to RAG:

- `<HF_USERNAME_team_member_RAG>/<RAG_DOCUMENT_REPO_NAME>`, replace `<RAG_DOCUMENT_REPO_NAME>` with the actual Huggingface Hub repo name of your submitted RAG documents.
- `<HF_USERNAME_team_member_RAG>/MNLP_M3_document_encoder`

4. `pdf` should be filled by you with your final report PDF (titled `<YOUR-GROUP-NAME>.pdf`). This directory should then have only one PDF.
5. `data` contains `data_repo.json`. In this file, you need to change `<HF_USERNAME_team_member_X>` to your Huggingface Hub username. Make sure that you have submitted the training data for your 4 models to the correct Huggingface Hub repositories, adhering to the following name convention:

- `<HF_USERNAME_team_member_DPO>/MNLP_M3_dpo_dataset`
- `<HF_USERNAME_team_member_MCQA>/MNLP_M3_mcqa_dataset`
- `<HF_USERNAME_team_member_QUANTIZED>/MNLP_M3_quantized_dataset`
- `<HF_USERNAME_team_member_RAG>/MNLP_M3_rag_dataset`

6. The `code` must contain the following:

- Four Bash Training Scripts:
You must provide four executable Bash scripts (.sh files) in the root of the `code` directory. These scripts are essential for reproducing your results and obtaining models equivalent to those you submit.

    - `train_dpo.sh`
    - `train_mcqa.sh`
    - `train_quantized.sh`
    - `train_rag.sh` 

- Four Corresponding Subfolders for Training Code:
For each of the four models, you should create a dedicated subfolder within code/ to house its specific training code. These folders should have the following structure:

    - `train_dpo/`
    - `train_mcqa/`
    - `train_quantized/`
    - `train_rag/ `

    By running these scripts we should be able to reproduce your training process and obtain models that are functionally equivalent to the ones you have developed and submitted.

## Validating Your Submission
**After you push your model weights and RAG documents to the correct Huggingface Hub repositories, make sure to test your models with the official [evaluation suite](https://github.com/eric11eca/lighteval-epfl-mnlp) in a fresh and clean environment (not the same environment you used for development).**

**If you got an error in the clean environment, you are responsible for debugging and correcting it.**
