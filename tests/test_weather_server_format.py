from __future__ import annotations

import pytest

pytest.importorskip("httpx")
pytest.importorskip("mcp")

from tools import weather_server


def test_format_weather_uses_returned_units() -> None:
    payload = {
        "location": {"name": "Boston", "country": "United States"},
        "current": {
            "time": "2026-03-12T09:00",
            "temperature_2m": 0.6,
            "relative_humidity_2m": 87,
            "wind_speed_10m": 4.6,
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "m/s",
        },
    }

    result = weather_server.format_weather(payload)

    assert "Temperature: 0.6°C" in result
    assert "Humidity: 87%" in result
    assert "Wind Speed: 4.6 m/s" in result


def test_format_weather_daily_uses_daily_units() -> None:
    payload = {
        "location": {"name": "Boston", "country": "United States"},
        "current": {
            "time": "2026-03-12T09:00",
            "temperature_2m": 0.6,
            "relative_humidity_2m": 87,
            "wind_speed_10m": 4.6,
        },
        "daily": {
            "time": ["2026-03-12"],
            "temperature_2m_max": [7.0],
            "temperature_2m_min": [-1.5],
            "precipitation_sum": [0.2],
        },
        "daily_units": {
            "temperature_2m_max": "°C",
            "precipitation_sum": "mm",
        },
    }

    result = weather_server.format_weather(payload)

    assert "max 7.0°C, min -1.5°C" in result
    assert "precipitation 0.2 mm" in result


def test_format_weather_includes_feels_like_when_available() -> None:
    payload = {
        "location": {"name": "Boston", "country": "United States"},
        "current": {
            "time": "2026-03-12T09:00",
            "temperature_2m": 0.6,
            "apparent_temperature": -2.1,
            "relative_humidity_2m": 87,
            "wind_speed_10m": 4.6,
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "m/s",
        },
    }

    result = weather_server.format_weather(payload)

    assert "Feels Like: -2.1°C" in result
