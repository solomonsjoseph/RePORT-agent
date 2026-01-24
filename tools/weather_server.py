from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("WeatherServer")

OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
USER_AGENT = "weather-app/1.0"


async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    Fetch weather data from OpenWeather.
    :param city: city name in English (e.g., "Beijing")
    :return: weather data dict or an error payload
    """
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "en",
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                OPENWEATHER_API_BASE,
                params=params,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            return {"error": f"HTTP error: {exc.response.status_code}"}
        except Exception as exc:
            return {"error": f"Request failed: {exc}"}


def format_weather(data: dict[str, Any] | str) -> str:
    """
    Format weather data into readable text.
    :param data: weather data dict or JSON string
    :return: formatted weather report
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as exc:
            return f"Unable to parse weather data: {exc}"

    if "error" in data:
        return f"Warning: {data['error']}"

    city = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "Unknown")
    temp = data.get("main", {}).get("temp", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "Unknown")

    return (
        f"{city}, {country}\n"
        f"Temperature: {temp}°C\n"
        f"Humidity: {humidity}%\n"
        f"Wind Speed: {wind_speed} m/s\n"
        f"Conditions: {description}\n"
    )


@mcp.tool()
async def query_weather(city: str) -> str:
    """
    Return today's weather for the specified city (English name required).
    :param city: city name (English)
    :return: formatted weather info
    """
    data = await fetch_weather(city)
    return format_weather(data)


@mcp.tool()
async def get_weather_tips(season: str) -> str:
    """
    Return season-specific weather tips.
    :param season: season name (spring, summer, autumn, winter)
    """
    tips = {
        "spring": "Spring is windy. Dress in layers and watch for allergies.",
        "summer": "Summer is hot. Stay hydrated and avoid heat exposure.",
        "autumn": "Autumn is dry. Moisturize and watch temperature shifts.",
        "winter": "Winter is cold. Keep warm and protect against colds.",
    }
    return tips.get(season.lower(), "Unknown season. Stay healthy.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
