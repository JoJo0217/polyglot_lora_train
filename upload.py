from huggingface_hub import HfApi
api = HfApi()
from huggingface_hub import login
login()
from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="",
    repo_id="",
    repo_type="model",
)