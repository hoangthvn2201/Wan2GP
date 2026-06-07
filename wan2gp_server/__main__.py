"""Run the server:  python -m wan2gp_server [--host H] [--port P] [--eager]"""

import argparse
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wan2gp_server",
        description="REST API server exposing WanGP text-to-image / text-to-video / image-to-video",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: WAN2GP_SERVER_HOST or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: WAN2GP_SERVER_PORT or 8000)")
    parser.add_argument("--eager", action="store_true", help="Load the WanGP runtime at startup instead of on first job")
    args = parser.parse_args()

    if args.host:
        os.environ["WAN2GP_SERVER_HOST"] = args.host
    if args.port:
        os.environ["WAN2GP_SERVER_PORT"] = str(args.port)
    if args.eager:
        os.environ["WAN2GP_SERVER_EAGER_INIT"] = "1"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from .app import create_app
    from .config import ServerConfig

    config = ServerConfig.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
