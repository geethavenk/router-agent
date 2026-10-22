import logging
import sys

from dotenv import load_dotenv

# Load environment
load_dotenv()

from agent import graph, CONFIDENCE_THRESHOLD

# Logging configuration
logging.basicConfig(
    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),

    handlers=[
        logging.StreamHandler(sys.stdout),

        logging.FileHandler(
            "router_agent.log",
            encoding="utf-8"
        )
    ]
)

logger = logging.getLogger("router_agent")

# Run agent

def ask_agent(query: str):
    logger.info("="*70)

    logger.info("Starting new request")

    logger.info("User query: %s", query)

    try:
        result = graph.invoke(
            {
                "query": query,
                "route": "",
                "confidence": 0.0,
                "final_route": "",
                "response": ""
            }
        )

        # Extract routing information

        predicted_route = result["route"]
        confidence = result["confidence"]
        final_route = result["final_route"]

        if final_route != predicted_route:
            logger.warning("Low confidence fallback | predicted=%s | confidence=%.2f | threshold=%.2f",
                           predicted_route, confidence, CONFIDENCE_THRESHOLD)
        # Logging
        logger.info("Router prediction: %s", predicted_route)

        logger.info("Router confidence: %.2f",confidence)

        logger.info("Final route: %s",final_route)

        logger.info("Request completed successfully")

        # Display result

        print("\n" + "=" * 70)

        print("USER QUERY")
        print(query)

        print("\nROUTER PREDICTION")
        print(predicted_route)

        print("\nCONFIDENCE")
        print(f"{confidence:.2f}")

        print("\nCONFIDENCE THRESHOLD")
        print(f"{CONFIDENCE_THRESHOLD:.2f}")

        print("\nFINAL ROUTE")
        print(final_route)

        print("\nRESPONSE")
        print(result["response"])

        print("=" * 70)


        return result

    except Exception as e:
        logger.exception("Error while processing request.")

        print("\nAn error occurred while processing your request.")

        return None


# Demo queries
if __name__ == "__main__":

    queries = [

        # ====================================================
        # TECHNICAL
        # ====================================================

        "The video player keeps buffering.",

        "I cannot log into my account.",


        # ====================================================
        # COURSE
        # ====================================================

        "I want to learn Generative AI. "
        "Which course should I start?",

        "Which course should I take to become "
        "a data scientist?",


        # ====================================================
        # BILLING
        # ====================================================

        "My payment was deducted but my "
        "course hasn't been activated.",

        "I was charged twice for my subscription.",


        # ====================================================
        # GENERAL
        # ====================================================

        "How can I change the name on my profile?",

        "Where can I find my certificates?",

        ## Vague question

        "It's not working"
    ]


    # ========================================================
    # Run all demo queries
    # ========================================================

    for query in queries:

        ask_agent(query)