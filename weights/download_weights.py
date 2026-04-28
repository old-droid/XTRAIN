from huggingface_hub import hf_hub_download
import os

def download_model():
    repo_id = "Syphaxtwt/ChapatiLMV"
    filename = "train_mv_weights.npz"
    
    # Define the target directory
    target_dir = os.path.join(os.getcwd(), "weights")
    
    # Create the 'weights' folder if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    print(f"Downloading {filename} from {repo_id}...")
    
    # Download to Hugging Face cache first
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # Move/Copy to your specific weights folder
    destination = os.path.join(target_dir, filename)
    
    import shutil
    shutil.copy(cached_path, destination)
    
    print(f"Success! Weights are now at: {destination}")

if __name__ == "__main__":
    download_model()
