import argparse
from test import HandGestureRecognizer, WebcamApp, ConfigLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hand-Recognition")
    parser.add_argument('--config', type=str, default="src/config/config.yaml",
                        help='Path to configuration file')
    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load(args.config)

    # Tạo recognizer và app
    recognizer = HandGestureRecognizer(config)
    app = WebcamApp(recognizer)
    app.run()
