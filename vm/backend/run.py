import uvicorn

if __name__ == '__main__':
    uvicorn.run("main:app",
                host="10.216.0.131",
                port=8001,
                reload=True,
                ssl_keyfile="/root/key.pem", 
                ssl_certfile="/root/cert.pem"
                )