import requests
import os
from app import settings
from PyPDF2 import PdfReader
import io


DOWNLOAD_PATH = os.path.join(settings.root, "downloads")
if not os.path.exists(DOWNLOAD_PATH):
	os.makedirs(DOWNLOAD_PATH)


def is_valid_pdf(data: bytes) -> bool:
	try:
		PdfReader(io.BytesIO(data))
		return True
	except Exception:
		return False


def get_pdf(url: str) -> bytes:
	# delete all existing pdfs in DOWNLOAD_PATH
	for file in os.listdir(DOWNLOAD_PATH):
		os.remove(os.path.join(DOWNLOAD_PATH, file))
	
	# get pdfs
	file_name = url.split("/")[-1]
	file_path = os.path.join(DOWNLOAD_PATH, file_name)
	response = requests.get(url)
	with open(file_path, "wb") as f:
		f.write(response.content)
	if not is_valid_pdf(response.content):
		os.remove(file_path)
		return None
	return file_path
