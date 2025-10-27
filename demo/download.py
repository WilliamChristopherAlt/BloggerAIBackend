from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    local_dir=r"modern_issues_demo/models/Mistral-7B-Instruct-v0.2-GGUF",
    local_dir_use_symlinks=False,
    resume_download=True,  # crucial: continues from partial download
    max_workers=8  # parallel chunk download
)
