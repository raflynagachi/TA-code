from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/files")
async def UploadImage(file: bytes = File(...)):
    with open('image.jpg', 'wb') as image:
        image.write(file)
        image.close()
    return 'got it'
