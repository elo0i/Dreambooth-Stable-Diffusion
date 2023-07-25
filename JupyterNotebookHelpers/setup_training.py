import argparse
import os
import sys
import shutil
from git import Repo
sys.path.append('/workspace/Drembooth-Stable-Diffusion/dreambooth_helpers')
os.chdir('/workspace/Dreambooth-Stable-Diffusion/')
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1
os.chdir('/workspace/Dreambooth-Stable-Diffusion/JupyterNotebookHelpers/')
from download_model import SDModelOption

class SetupTraining:
    def __init__(self, args):
        self.training_images_save_path = "./training_images"
        self.config_save_path = "./joepenna-dreambooth-configs"
        self.selected_model: SDModelOption = None
        self.args = args

    def submit_form_click(self, b):
        # training images
        uploaded_training_images = self.args.training_images
        if len(uploaded_training_images) == 0:
            print("No training images provided, please provide a path to the training images.", file=sys.stderr)
            return
        else:
            self.handle_training_images(uploaded_training_images)

        # Regularization Images
        regularization_images_dataset = self.args.reg_images_select
        regularization_images_folder_path = self.download_regularization_images(regularization_images_dataset)

        config = JoePennaDreamboothConfigSchemaV1()
        config.saturate(
            project_name=self.args.project_name,
            max_training_steps=int(self.args.max_training_steps),
            save_every_x_steps=int(self.args.save_every_x_steps),
            training_images_folder_path=self.training_images_save_path,
            regularization_images_folder_path=regularization_images_folder_path,
            token=self.args.token,
            token_only=False,
            class_word=self.args.class_word,
            flip_percent=float(self.args.flip),
            learning_rate=self.args.learning_rate,
            model_repo_id=self.selected_model.repo_id,
            model_path=self.selected_model.filename,
            run_seed_everything=False,
        )

        config.save_config_to_file(
            save_path=self.config_save_path,
            create_active_config=True
        )

    def download_regularization_images(self, dataset) -> str:
        # Download Regularization Images
        repo_name = f"Stable-Diffusion-Regularization-Images-{dataset}"
        path_to_reg_images = os.path.join(repo_name, dataset)

        if not os.path.exists(path_to_reg_images):
            print(f"Downloading regularization images for {dataset}. Please wait...")
            Repo.clone_from(f"https://github.com/djbielejeski/{repo_name}.git", repo_name)
            print(f"✅ Regularization images for {dataset} downloaded successfully.")
        else:
            print(f"✅ Regularization images for {dataset} already exist. Skipping download...")

        return path_to_reg_images

    def handle_training_images(self, uploaded_images):
        print("Copying training images...")
        if os.path.exists(self.training_images_save_path):
            # remove existing images
            shutil.rmtree(self.training_images_save_path)

        # Copy the training images
        shutil.copytree(uploaded_images, self.training_images_save_path)

        print(f"✅ Training images copied successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_images", type=str, required=True)
    parser.add_argument("--reg_images_select", type=str, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--max_training_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--class_word", type=str, required=True)
    parser.add_argument("--flip", type=float, required=True)
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--save_every_x_steps", type=int, required=True)
    args = parser.parse_args()
    setup = SetupTraining(args)
    setup.submit_form_click(None)
