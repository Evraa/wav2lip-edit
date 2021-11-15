import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

print ("Pick one of these two models:\n0-wav2lip_gan.pth\n1-wav2lip.pth")
ans = int (input(">>"))
if ans == 0:
    id = '1jQOJInh8cDj2mrbUgcQxhCc7rpAgyV1-'
    destination = 'Wav2lip/checkpoints/wav2lip_gan.pth'
elif ans == 1:
    id = '1ws1Ftl2nMMjRp7kEbnb9th6CdHTIsHjK'
    destination = "Wav2lip/checkpoints/wav2lip.pth"
download_file_from_google_drive(id, destination)
