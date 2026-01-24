from __future__ import annotations

import json
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "weather-app/1.0"


async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    Fetch weather data from OpenWeather.
    :param city: city name in English (e.g., "Boston")
    :return: weather data dict or an error payload
    """
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            geo_response = await client.get(
                OPEN_METEO_GEOCODE,
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                headers=headers,
                timeout=30.0,
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            results = geo_data.get("results") or []
            if not results:
                return {"error": "No matching city found."}
            location = results[0]
            forecast_response = await client.get(
                OPEN_METEO_FORECAST,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                },
                headers=headers,
                timeout=30.0,
            )
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            return {
                "location": location,
                "current": forecast_data.get("current", {}),
            }
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

    location = data.get("location", {})
    current = data.get("current", {})
    city = location.get("name", "Unknown")
    country = location.get("country", "Unknown")
    temp = current.get("temperature_2m", "N/A")
    humidity = current.get("relative_humidity_2m", "N/A")
    wind_speed = current.get("wind_speed_10m", "N/A")

    return (
        f"{city}, {country}\n"
        f"Temperature: {temp}°C\n"
        f"Humidity: {humidity}%\n"
        f"Wind Speed: {wind_speed} m/s\n"
    )


@mcp.tool()
async def query_weather(city: str) -> str:
    """
    Return today's weather for the specified city
    :param city: city name 
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
