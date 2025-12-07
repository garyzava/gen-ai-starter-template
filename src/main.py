from src.config.settings import settings


def main():
    # 1. Access standard variables
    print(f"Starting {settings.APP_NAME} in {settings.ENVIRONMENT} mode...")

    # 2. Access Secrets
    # Note: You must use .get_secret_value() to reveal the string.
    # Printing settings.OPENAI_API_KEY directly will show '**********'
    _ = settings.OPENAI_API_KEY.get_secret_value()  # Validate key exists

    print(f"Using Model: {settings.LLM_MODEL}")
    print(f"Temp: {settings.LLM_TEMPERATURE}")

    # 3. Path objects are already standard python Pathlibs
    if not settings.VECTOR_DB_PATH.exists():
        settings.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Created database at: {settings.VECTOR_DB_PATH}")

if __name__ == "__main__":
    main()
