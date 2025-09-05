from huggingface_hub import list_repo_files

# Check what files are in your repo
files = list_repo_files("Dinara777/epidemic_classificator_model")
print("Files in repository:", files)
