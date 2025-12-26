import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from server.server import FallDetectionServer
from config.settings import Settings


def main():
    settings = Settings()

    server = FallDetectionServer(
        camera_id=settings.CAMERA_ID,
        host=settings.HOST,
        port=settings.PORT
    )

    server.run()


if __name__ == "__main__":
    main()
