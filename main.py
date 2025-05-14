import logging
from dotenv import load_dotenv
from typing import Optional

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("saudia-air-agent")
logger.setLevel(logging.INFO)

load_dotenv()


class SaudiaReservationAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an airline reservation agent for Saudia Airways. You are only allowed to answer travel-related queries. "
                "Always stay focused on the user's itinerary and respond in a crisp, humble, polite, and professional manner. "
                "Your responses should never include tabular formats, new-line characters, or markdown. "
                "Keep each response short and suitable for reading out loud on a phone call. All critical facts must be summarized in under 3 sentences.\n\n"

                "Behavioral Instructions:\n"
                "- Start every conversation by calling the `current_time` function to get the current date and time.\n"
                "- Never refer to or allow booking for past dates.\n"
                "- If user gives multiple requests, complete them one by one in sequence.\n"
                "- If number of passengers isn’t provided, assume 1. Do not ask.\n"
                "- Maintain a consistent order ID through the conversation until payment is completed.\n"
                "- For group bookings, ensure responses reference the group.\n"
                "- Always mention flight pricing.\n"
                "- Assume direct return flights unless stopover is specified.\n"
                "- If travel date is not given, ask for it. Never assume today.\n"
                "- Use `current_time` to resolve relative dates like today, tomorrow, next week — always toward the future.\n"
                "- If year is missing in a date, assume it’s in the future, not the past. Current year is 2025.\n"
                "- If pickup or drop-off is at an airport, use the airport name of the city, not full address.\n"
                "- Only allow cab bookings for intracity trips.\n"
                "- Store hotel options unless user asks to change them.\n"
                "- If flight is selected, hotel check-in and check-out must match the arrival and departure dates — do not allow hotel booking outside this range.\n"
                "- If user asks to optimize the order, show allowed changes: cheaper hotels, cabs, or lower class flights. Do not go below guest class.\n"
                "- After optimization, use the reviewOrder tool to confirm everything again.\n"
                "- Keep hotel search ID consistent unless the user asks for a different itinerary.\n"
                "- Maintain the language used by the user.\n\n"

                "Tool Usage Instructions:\n"
                "- Use the `book_flight` tool to confirm flight booking once the user provides origin, destination, and departure date. Always include class and passenger count if available.\n"
                "- Use the `book_hotel` tool only after a flight is selected. The check-in date must match the flight arrival and the check-out date must match the return flight date or trip end.\n"
                "- Use the `book_cab` tool for intracity transportation needs. Only book cabs for travel within the same city, not between cities.\n"
                "- Use the `select_meal` tool after flight booking is confirmed to set meal preferences for the flight. Offer this option for flights longer than 2 hours.\n"
                "- Use the `process_payment` tool after all bookings are confirmed to complete the transaction. Always verify booking details before processing payment.\n"
                "- Respond with a brief confirmation including booking ID for flight, hotel, and cab bookings.\n\n"

                "Always be concise, clear, and easy to follow on voice. Never go beyond 3 sentences when sharing any information, especially flight or booking details."
            )
        )

    @function_tool()
    async def current_time(self):
        """Returns the current date and time. Always call this at the beginning of the conversation to determine the date context."""
        import datetime
        now = datetime.datetime.utcnow()
        logger.info(f"Current UTC time is {now.isoformat()}")
        return now.isoformat()

    @function_tool()
    async def book_flight(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        passenger_count: int = 1,
        class_type: Optional[str] = "economy"
    ):
        """Books a mock flight and returns flight summary."""
        class_type = class_type or "economy"
        flight_id = f"FL-{origin[:3].upper()}{destination[:3].upper()}123"
        summary = (
            f"Flight booked from {origin} to {destination} on {departure_date} "
            f"for {passenger_count} passenger(s) in {class_type} class. "
        )
        if return_date:
            summary += f"Return flight on {return_date}. "
        summary += f"Booking ID is {flight_id}."
        logger.info(summary)
        return {
            "flight_id": flight_id,
            "status": "confirmed",
            "summary": summary
        }

    @function_tool()
    async def book_hotel(
        self,
        city: str,
        check_in_date: str,
        check_out_date: str,
        guests: int = 1,
        hotel_type: Optional[str] = "3-star"
    ):
        """Books a mock hotel and returns hotel summary."""
        hotel_type = hotel_type or "3-star"
        hotel_id = f"HT-{city[:3].upper()}567"
        summary = (
            f"{hotel_type.capitalize()} hotel booked in {city} from {check_in_date} to {check_out_date} "
            f"for {guests} guest(s). Booking ID is {hotel_id}."
        )
        logger.info(summary)
        return {
            "hotel_id": hotel_id,
            "status": "confirmed",
            "summary": summary
        }
        
    @function_tool()
    async def book_cab(
        self,
        city: str,
        pickup_location: str,
        dropoff_location: str,
        pickup_time: str,
        passengers: int = 1,
        cab_type: Optional[str] = "standard"
    ):
        """Books a mock cab for intracity travel and returns cab booking summary."""
        cab_type = cab_type or "standard"
        cab_id = f"CB-{city[:3].upper()}{pickup_time[-4:].replace(':', '')}"
        summary = (
            f"{cab_type.capitalize()} cab booked in {city} from {pickup_location} to {dropoff_location} "
            f"at {pickup_time} for {passengers} passenger(s). Booking ID is {cab_id}."
        )
        logger.info(summary)
        return {
            "cab_id": cab_id,
            "status": "confirmed",
            "summary": summary
        }
        
    @function_tool()
    async def select_meal(
        self,
        flight_id: str,
        meal_type: str,
        dietary_restrictions: Optional[str] = None,
        special_requests: Optional[str] = None,
        passenger_name: Optional[str] = None
    ):
        """Selects meal preferences for a flight and returns confirmation."""
        meal_pref_id = f"MP-{flight_id[-3:]}{hash(meal_type)%1000:03d}"
        
        meal_summary = f"{meal_type} meal"
        if dietary_restrictions:
            meal_summary += f" ({dietary_restrictions})"
        
        passenger_info = ""
        if passenger_name:
            passenger_info = f" for {passenger_name}"
        
        summary = f"Meal preference set to {meal_summary} for flight {flight_id}{passenger_info}. Preference ID: {meal_pref_id}."
        
        if special_requests:
            summary += f" Special request noted: {special_requests}."
            
        logger.info(summary)
        return {
            "meal_preference_id": meal_pref_id,
            "flight_id": flight_id,
            "meal_type": meal_type,
            "dietary_restrictions": dietary_restrictions,
            "special_requests": special_requests,
            "status": "confirmed",
            "summary": summary
        }
    
    @function_tool()
    async def process_payment(
        self,
        booking_ids: list[str],
        payment_method: str,
        amount: float,
        currency: str = "SAR",
        customer_name: Optional[str] = None,
        email: Optional[str] = None
    ):
        """Processes payment for bookings and returns payment confirmation."""
        payment_id = f"PY-{hash(str(booking_ids))%10000:04d}"
        booking_ids_str = ", ".join(booking_ids)
        
        summary = (
            f"Payment of {amount} {currency} processed successfully via {payment_method} "
            f"for booking(s): {booking_ids_str}. Payment ID: {payment_id}."
        )
        
        if customer_name:
            customer_info = f" for {customer_name}"
            if email:
                customer_info += f" ({email})"
            summary += customer_info
            
        logger.info(summary)
        return {
            "payment_id": payment_id,
            "status": "confirmed",
            "amount": amount,
            "currency": currency,
            "booking_ids": booking_ids,
            "summary": summary
        }


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await ctx.wait_for_participant()
    await session.start(agent=SaudiaReservationAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
