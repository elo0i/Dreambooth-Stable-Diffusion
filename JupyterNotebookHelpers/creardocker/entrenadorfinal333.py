import os
import subprocess
import time
import argparse

parser = argparse.ArgumentParser(description='Entrenador personalizado')
parser.add_argument('--user_id', type=str, required=True, help='ID de usuario')
parser.add_argument('--order_id', type=str, required=True, help='ID de pedido')
args = parser.parse_args()

user_id = args.user_id
print("El user_id es:", user_id)

order_id = args.order_id
print("El order_id es:", order_id)

instance_id = os.environ['VAST_CONTAINERLABEL']
print("El instance_id es:", instance_id)

instance_id = instance_id.replace("C.", "")
print("El instance_id sin 'C.' es:", instance_id)


# Descarga el repositorio de Dreambooth-Stable-Diffusion
subprocess.run(["git", "clone", "https://github.com/elo0i/Dreambooth-Stable-Diffusion"])

#Instalar .txt requerimientos
#subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

required_packages = [
    "numpy==1.23.1",
    "pytorch-lightning==1.7.6",
    "csv-logger",
    "torchmetrics==0.11.1",
    "torch-fidelity==0.3.0",
    "albumentations==1.1.0",
    "opencv-python==4.7.0.72",
    "pudb==2019.2",
    "omegaconf==2.1.1",
    "pillow==9.4.0",
    "einops==0.4.1",
    "transformers==4.25.1",
    "kornia==0.6.7",
    "diffusers[training]==0.3.0",
    "captionizer==1.0.1",
    "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
    "git+https://github.com/openai/CLIP.git@main#egg=clip",
    "huggingface_hub",
    "gitpython",
    "google-cloud-storage",
    "requests",
    "argparse",
]

for package in required_packages:
    if "git+" in package:
        subprocess.check_call(["pip", "install", "-e", package.replace("-e ", "")])
    else:
        subprocess.check_call(["pip", "install", package])



import shutil
#from IPython.display import clear_output
#from huggingface_hub import hf_hub_download
from google.cloud import storage

# Crear la carpeta 'training_images' si no existe
training_images_directory = "Dreambooth-Stable-Diffusion/training_images"
os.makedirs(training_images_directory, exist_ok=True)

# Configura las credenciales para Google Cloud Storage (reemplaza 'path/to/credentials.json' con el archivo de credenciales correcto)
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

import json
import tempfile

credentials_content = r"""
{
  "type": "service_account",
  "project_id": "entrenadoria",
  "private_key_id": "864e35bb90713d9aa6d6a529318358c40fb12fda",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCuPfEGyatN+09e\nUsnG68KGomBnhNG4Jv/jZyTD6KSgRBMRpPs9G1YaoqR8zyP8H6K7mLeZLSCPHGRB\nJihNBEPytkX5bCpmVFxhNwPtZCzMVUVQhmNWBk01Vwty5DGZAJfluidKB9w3Ug/E\nMZh3C/eJDcxjzQ+l8V6DdcE1inCoSoZ3/fPWsDybL3yCkvIrsts00m/wvl8xBnEv\nrPy9TWOtt42Gb7X/yccPQVEfLckSngmATCau9w6Jmr3D/SlnphE7IWVfpOtw6tdf\nOV+QWDnIaWDK6pc6S5TUsnAvLvX+t8aeioxJ2sjQ42xs0Y+EPCTrtWd1/zRi3/5D\n5WuXLWIzAgMBAAECggEAMCjKr7eZ84ncnBOB1ctgDtpejv41ARM2cuIVVVi999YY\n9y2Ei5U8rUv67sxKA+uyjOtfA6VndGbChwdG6FKffTxIBvKQnYv/pJcSLNEdWLTQ\n3brnReWj/XQ7o9vSoZl6YnKbXWjiwx+ZX/7dRzo1htobfhI1mwYlu2wWpPfIv5qz\niFuRacpO/yKe8JgzINU2Cp1Oag3WjaBgjEJ5/QO7HthCEg/t++3hWCOrafiALgyo\n5+y1cIJoSt1jFQRmw7rC5kT/eP+VTq9FfcdCoAzaxmFThVUCY+C2gihioyC1mswq\n1ZnnzcNZhJ2mNGwmUVn6PorRjjSCm85Yi5+Mc96+gQKBgQD18z6QmGVyzrLY3S8q\n6wvnlIFLwVC3frn55Lu8dqZaFZFgmYMNy8mrNvb/FLKZkXkt7ZbSUP+OkVTguxKs\nl6Z/xG8GEk2QiFVUX81F2g1B6B9Vi7Arm+d1qsBbgoE+Xc9yZRsvfLySWM3rw/vZ\nonpL4lfpiL8zTYCv57HUycEKoQKBgQC1XJhbOZqggsaqU+mltxRa02bc6K0oreBK\nrk8lNHQZ7HIG52mlaCbtbsoWJs3qleg+PUuur6fzGu5MQOrF9inQ7uk9Nsx7lF+F\nskZ5M9O0athOJFcdHPK3In/K2iN9Lr8mxt1hRZjlF+iTxBPiSz1YXXCZE0ta/ckS\nA49PL6DwUwKBgA3d5AYlAXtCmiTN+63QyMAKyGtr/9AIrhWfxtHuYpyroKGwpgnu\nFnW3yJ9DHHq6D/n97kX3WSFBomZ1Ra1Dc5i6i4PtHkBq31y9dgZdL+gqXTHmiU08\nIgWpVeUS65SHl32co7a+sqcRqLKFPzrbBUgn/8rj8dvDn+DLEGSt51thAoGBAIXV\n9gzj/orS1x6c3ABRkbDQ7si44Af0AF+8MGXJRqBWz3Lu1RSePpPavUEJk824oHFF\ntJMNx4fsaMxW36oE1aj8lZx50v6jaLZ17/HDYEh0zHkl8i9mzGp/CAU/Yw8fLyrD\neF0vCfyN3zEkcnP9iCCsm8oq5eIZBIfJnrkV8dNHAoGADgTSKsBun6kUaTi4Ujvu\n3gZ7Vu8rCLLEZjHngaTEnW5pRoCGHW8pMR7/I+AzrlplYykY1uCwaX8HY1Y3/tn5\nFF4ey83pBaQSB2TKTyNXgzqvrtVCTIJYczmuSmcrVP68WptEJEMF+XQuB5o5gbKa\nYUANBfk6nGqNpMZS3c/Q2cI=\n-----END PRIVATE KEY-----\n",
  "client_email": "entrenador-ia@entrenadoria.iam.gserviceaccount.com",
  "client_id": "108033510327501318153",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/entrenador-ia%40entrenadoria.iam.gserviceaccount.com"
}
"""

# Carga la cadena JSON directamente
credentials = json.loads(credentials_content)

with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
    # Escribe las credenciales en el archivo temporal sin usar json.dump()
    temp.write(credentials_content)
    temp.flush()
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp.name


#user_id = "anujbostdata"  # Reemplaza esto con el ID del usuario actual

# Descarga imágenes de entrenamiento de Google Cloud Storage
bucket_name = "bucket-entrenamiento"
prefix = f"{user_id}/{order_id}/training_data/"
training_images_directory = "Dreambooth-Stable-Diffusion/training_images"

client = storage.Client()
bucket = client.get_bucket(bucket_name)
blobs = bucket.list_blobs(prefix=prefix)

os.makedirs(training_images_directory, exist_ok=True)

for blob in blobs:
    file_name = os.path.basename(blob.name)
    print(f"Nombre del blob: {blob.name}, Nombre del archivo: {file_name}")  # Agregar esta línea para depurar
    if file_name:  # Verifica si file_name no está vacío
        file_path = os.path.join(training_images_directory, file_name)
        blob.download_to_filename(file_path)
    else:
        print(f"Advertencia: Se encontró un archivo sin nombre o una carpeta vacía: {blob.name}")


import requests

url = "https://huggingface.co/panopstor/EveryDream/resolve/main/sd_v1-5_vae.ckpt"
response = requests.get(url, stream=True)
response.raise_for_status()

with open("model.ckpt", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

downloaded_model_path = "model.ckpt"


# Agregar una pausa de 5 segundos para que el explorador de archivos pueda encontrar el modelo
print("inicio pausa***********")
time.sleep(3)
print("fin pausa**********************************************")

##MÉTODO NOTEBOOK
# Move the sd_v1-5_vae.ckpt to the root of this directory as "model.ckpt"
#actual_locations_of_model_blob = !readlink -f {downloaded_model_path}
#!mv {actual_locations_of_model_blob[-1]} model.ckpt
#clear_output()
#print("✅ model.ckpt successfully downloaded")

# Encuentra la ruta absoluta del archivo descargado
actual_locations_of_model_blob = os.path.abspath(downloaded_model_path)

# Mueve el archivo sd_v1-5_vae.ckpt a la raíz del directorio como "model.ckpt"
shutil.move(actual_locations_of_model_blob, "Dreambooth-Stable-Diffusion/model.ckpt")


print("✅ model.ckpt successfully downloaded")

# Descarga las imágenes de regularización
dataset = "person_ddim"
subprocess.run(["git", "clone", f"https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-{dataset}.git"])

# Agregar una pausa de 3 segundos para que el explorador de archivos pueda encontrar las imágenes
print("inicio pausa2***********")
time.sleep(3)
print("fin pausa2**********************************************")

# Mueve las imágenes de regularización a la ubicación correcta
shutil.move(f"Stable-Diffusion-Regularization-Images-{dataset}/{dataset}", f"Dreambooth-Stable-Diffusion/regularization_images/{dataset}")

# Configuración del entrenamiento
project_name = str(user_id)
max_training_steps = 4000
class_word = "person"
i_am_training_a_persons_face = True
flip_p_arg = 0.0 if i_am_training_a_persons_face else 0.5
token = "TMF"
save_every_x_steps = 500
reg_data_root = f"/workspace/Dreambooth-Stable-Diffusion/regularization_images/{dataset}"

# Elimina .ipynb_checkpoints si existen
shutil.rmtree("Dreambooth-Stable-Diffusion/training_images/.ipynb_checkpoints", ignore_errors=True)

print("EMPEZANDO entrenamiento en 3 segundos")

time.sleep(3)

# Ejecuta el entrenamiento
os.system(f'python Dreambooth-Stable-Diffusion/main.py --project_name "{project_name}" --debug False --max_training_steps {max_training_steps} --token "{token}" --training_model "/workspace/Dreambooth-Stable-Diffusion/model.ckpt" --training_images "/workspace/Dreambooth-Stable-Diffusion/training_images" --regularization_images "{reg_data_root}" --class_word "{class_word}" --flip_p {flip_p_arg} --save_every_x_steps {save_every_x_steps}')

print("Entrenamiento FINALIZADO!!!")

#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

time.sleep(3)

import glob

def most_recent_model():
    list_of_files = glob.glob("/workspace/trained_models/*.ckpt")  # obtén solo los archivos .ckpt
    latest_file = max(list_of_files, key=os.path.getmtime)  # obtén el archivo más reciente basándote en la última hora de modificación
    return latest_file

def generate_images(prompts):
    original_dir = os.getcwd()  # obtén el directorio de trabajo actual
    try:
        os.chdir("/workspace/Dreambooth-Stable-Diffusion")  # cambia el directorio de trabajo
        for prompt in prompts:
            command = [
                'python', 'scripts/stable_txt2img.py',
                '--ddim_eta', '0.0',
                '--n_samples', '1',
                '--n_iter', '9',
                '--scale', '7.0',
                '--ddim_steps', '64',
                '--ckpt', most_recent_model(),
                '--prompt', prompt
            ]
            subprocess.check_call(command)
    finally:
        os.chdir(original_dir)  # regresa al directorio de trabajo original


import sys
sys.path.append('/workspace/Dreambooth-Stable-Diffusion')

# Definir tus prompts.
prompts = [
    "TMF person as a masterpiece portrait painting by John Singer Sargent in the style of Rembrandt",
    "TMF person sexy selfie, High angle photo of a gorgeous TMF person in the style of stefan kostic, realistic skin texture",
    "TMF person, nude, naked, sexy portrait, hooker, photobook"
]

# Ahora simplemente llama a la función en el punto de tu script donde file_name ya está definido
generate_images(prompts)

print("imagenes ejemplo YA generadas? Subiendo modelos al bucket")

#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

#from google.cloud import storage
#import os
import datetime

# Después de que el entrenamiento haya terminado, sube los modelos entrenados a Google Cloud Storage
trained_models_directory = "/workspace/trained_models"
destination_folder = f"{user_id}/{order_id}/trained_models/"

# Define la función para obtener los últimos 4 modelos
def last_4_models(directory):
    # Listamos todos los archivos ckpt en el directorio
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".ckpt")]

    # Ordenamos los archivos por fecha de modificación
    files.sort(key=lambda x: os.path.getmtime(x))

    # Devolvemos los últimos 4 archivos
    return files[-4:]

# Obtén los últimos 4 archivos
last_4_files = last_4_models(trained_models_directory)

for file in last_4_files:
    file_name = os.path.basename(file)
    destination_blob_name = os.path.join(destination_folder, file_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file)
    print(f"Archivo {file_name} subido a {destination_blob_name}.")
    # En el bucle donde se suben los archivos al bucket, añade lo siguiente para generar un enlace
    # al archivo con un tiempo de expiración de 1 hora (3600 segundos):
    url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(seconds=36000), method="GET")
    print(f"Enlace del archivo {file_name}: {url}")


#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

output_directory = "/workspace/Dreambooth-Stable-Diffusion/outputs/txt2img-samples"
destination_folder_images = f"{user_id}/{order_id}/output_images/"

def last_generated_images(directory):
    # Listamos todos los archivos png y jpg en el directorio
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png") or f.endswith(".jpg")]

    # Ordenamos los archivos por fecha de modificación
    files.sort(key=lambda x: os.path.getmtime(x))

    return files

# Obtén todas las imágenes generadas
last_generated_files = last_generated_images(output_directory)

for file in last_generated_files:
    file_name = os.path.basename(file)
    destination_blob_name = os.path.join(destination_folder_images, file_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file)
    print(f"Archivo {file_name} subido a {destination_blob_name}.")
    # En el bucle donde se suben los archivos al bucket, añade lo siguiente para generar un enlace
    # al archivo con un tiempo de expiración de 1 hora (3600 segundos):
    url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(seconds=36000), method="GET")
    print(f"Enlace del archivo {file_name}: {url}")


#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# Al final del script, apaga la instancia de vast.ai mediante mi API
print("Al parecer los modelos se han subido, parando instancia en 3, 2, 1...")
time.sleep(3)

response = requests.post('https://apigestioninstancias-4kqqx43hiq-no.a.run.app/training_completed', data={'instance_id': instance_id})
print("La respuesta de training completed es:", response.text)


