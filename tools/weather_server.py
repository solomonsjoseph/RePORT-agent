from __future__ import annotations

import json
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "weather-app/1.0"


def _normalize_date_range(
    start_date: str | None,
    end_date: str | None,
) -> tuple[str | None, str | None]:
    if start_date and not end_date:
        return start_date, start_date
    if end_date and not start_date:
        return end_date, end_date
    return start_date, end_date


async def fetch_weather(
    city: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any] | None:
    """
    Fetch weather data from OpenWeather.
    :param city: city name in English (e.g., "Boston")
    :param start_date: start date (YYYY-MM-DD)
    :param end_date: end date (YYYY-MM-DD)
    :return: weather data dict or an error payload
    """
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            geo_response = await client.get(
                OPEN_METEO_GEOCODE,
                params={"name": city, "count": 10, "language": "en", "format": "json"},
                headers=headers,
                timeout=30.0,
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            results = geo_data.get("results") or []
            if not results:
                return {"error": "No matching city found."}

            normalized_city = city.strip().lower()
            exact_name_matches = [
                r for r in results if str(r.get("name", "")).strip().lower() == normalized_city
            ]
            candidates = exact_name_matches or results
            location = max(candidates, key=lambda r: int(r.get("population") or 0))
            start_date, end_date = _normalize_date_range(start_date, end_date)
            forecast_params: dict[str, Any] = {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
                "temperature_unit": "celsius",
                "wind_speed_unit": "ms",
                "timezone": "auto",
            }
            if start_date and end_date:
                forecast_params.update(
                    {
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                )
            else:
                forecast_params.update(
                    {
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                        "forecast_days": 1,
                    }
                )
            forecast_response = await client.get(
                OPEN_METEO_FORECAST,
                params=forecast_params,
                headers=headers,
                timeout=30.0,
            )
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            return {
                "location": location,
                "current": forecast_data.get("current", {}),
                "current_units": forecast_data.get("current_units", {}),
                "daily": forecast_data.get("daily", {}),
                "daily_units": forecast_data.get("daily_units", {}),
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
    apparent_temp = current.get("apparent_temperature")
    current_units = data.get("current_units") or {}
    temp_unit = current_units.get("temperature_2m", "°C")
    humidity_unit = current_units.get("relative_humidity_2m", "%")
    wind_unit = current_units.get("wind_speed_10m", "m/s")
    observation_time = current.get("time", "N/A")
    daily = data.get("daily") or {}
    daily_units = data.get("daily_units") or {}
    daily_temp_unit = daily_units.get("temperature_2m_max", temp_unit)
    daily_precip_unit = daily_units.get("precipitation_sum", "mm")
    daily_dates = daily.get("time", [])
    daily_max = daily.get("temperature_2m_max", [])
    daily_min = daily.get("temperature_2m_min", [])
    daily_precip = daily.get("precipitation_sum", [])

    lines = [
        f"{city}, {country}\n"
        f"Observed: {observation_time}\n"
        f"Temperature: {temp}{temp_unit}\n"
        + (f"Feels Like: {apparent_temp}{temp_unit}\n" if apparent_temp not in (None, "N/A") else "")
        + f"Humidity: {humidity}{humidity_unit}\n"
        + f"Wind Speed: {wind_speed} {wind_unit}\n"
    ]
    if daily_dates:
        lines.append("\nForecast:\n")
        for idx, date in enumerate(daily_dates):
            max_temp = daily_max[idx] if idx < len(daily_max) else "N/A"
            min_temp = daily_min[idx] if idx < len(daily_min) else "N/A"
            precip = daily_precip[idx] if idx < len(daily_precip) else "N/A"
            lines.append(
                f"- {date}: max {max_temp}{daily_temp_unit}, min {min_temp}{daily_temp_unit}, "
                f"precipitation {precip} {daily_precip_unit}\n"
            )
    return "".join(lines)


@mcp.tool()
async def query_weather(
    city: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """
    Return weather for the specified city and optional date range (YYYY-MM-DD).
    :param city: city name 
    :param start_date: start date (YYYY-MM-DD)
    :param end_date: end date (YYYY-MM-DD)
    :return: formatted weather info
    """
    data = await fetch_weather(city, start_date=start_date, end_date=end_date)
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
