from fastapi import FastAPI
import uvicorn

app = FastAPI()

users = [{"id": 1, "name": "peter", "age": 22}, {"id": 2, "name": "frank", "age": 40}]

@app.get("/")
async def index():
    return {"message": "Hello from app!"}

@app.get("/users")
async def all_users():
    return users

@app.get("/users/{user_id}")
async def get_user(user_id):
    name = users[int(user_id)].get(" name")
    return {"message": f"Hello from {name}"}

if __name__ == "__main__":
    uvicorn.run('app:app', host="0.0.0.0", port = 8000, reload = True)