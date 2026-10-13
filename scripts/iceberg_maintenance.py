#!/usr/bin/env python3
"""Command-line entry point for Iceberg maintenance operations."""

from video_commerce.services.iceberg_maintenance.main import main


if __name__ == "__main__":
    raise SystemExit(main())
