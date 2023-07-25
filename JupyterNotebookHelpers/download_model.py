import argparse
from huggingface_hub import hf_hub_download

class SDModelOption:
    def __init__(self, repo_id, filename, manual=False):
        self.repo_id = repo_id
        self.filename = filename
        self.manual = manual

    def download(self):
        if self.is_valid():
            print(f"Downloading '{self.repo_id}/{self.filename}'")
            return hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename
            )
        else:
            raise Exception(
                f"Model not valid. repo_id: {self.repo_id} or filename: {self.filename} are missing or invalid.")

    def is_valid(self):
        return (self.repo_id is not None and self.repo_id != '') and \
            (self.filename is not None and self.filename != '' and '.ckpt' in self.filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    selected_model = SDModelOption(repo_id=args.repo_id, filename=args.filename)
    selected_model.download()
