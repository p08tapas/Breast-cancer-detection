"""
This represents the entrypoint for the mammogram inference app.
We need to run python.main.py to start the app.
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8080)
