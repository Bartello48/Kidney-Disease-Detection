from src.config_reader import ConfigReader


def main():
    print("Hello from kidney-disease-detection!")
    config = ConfigReader()
    config._restore_config()


if __name__ == "__main__":
    main()
