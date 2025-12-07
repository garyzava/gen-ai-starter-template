import asyncio
import time

from src.config.settings import settings
from src.llm.client import OpenAIClient
from src.schemas.chat import Message, Role


async def main():
    print(f"🔍 Starting System Check for: {settings.APP_NAME}")
    print(f"📂 Environment: {settings.ENVIRONMENT}")
    print(f"🤖 Model: {settings.LLM_MODEL}")
    print("-" * 50)

    # 1. Initialize Client
    try:
        client = OpenAIClient()
        print("✅ Client initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return

    # Prepare a test message
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful technical assistant."),
        Message(role=Role.USER, content="Explain the benefits of Pydantic in one sentence.")
    ]

    # 2. Test Standard Completion (achat)
    print("\n[Test 1] Standard Completion...")
    start_time = time.time()

    try:
        response = await client.achat(messages)
        duration = time.time() - start_time

        print(f"⏱️  Latency: {duration:.2f}s")
        print(f"💬 Response: {response.content}")
        print(f"🎫 Tokens: {response.token_usage}")
        print("✅ Standard Completion Passed.")

    except Exception as e:
        print(f"❌ Standard Completion Failed: {e}")

    # 3. Test Streaming (astream)
    print("\n[Test 2] Streaming Response...")
    print("💬 Stream: ", end="", flush=True)

    try:
        async for chunk in client.astream(messages):
            print(chunk, end="", flush=True)
            # simulate a tiny delay if you want to see the effect visually
            # await asyncio.sleep(0.01)

        print("\n✅ Streaming Passed.")

    except Exception as e:
        print(f"\n❌ Streaming Failed: {e}")

if __name__ == "__main__":
    # Asyncio entry point
    asyncio.run(main())
