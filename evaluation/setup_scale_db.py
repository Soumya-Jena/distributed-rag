from src.scale_db import DATABASE_URL_SCALE, ensure_scale_database


def main():
    ensure_scale_database()
    print(f"Isolated scalability database ready: {DATABASE_URL_SCALE}")


if __name__ == "__main__":
    main()

