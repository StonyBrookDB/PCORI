import uvicorn


def main() -> None:
    uvicorn.run(
        "app.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        env_file=".env",
    )


if __name__ == "__main__":
    main()
